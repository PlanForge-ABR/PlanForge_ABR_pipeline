"""
Zero-Shot DSPy Module - DSPy integration and evaluation orchestration.

This module contains the DSPy module, domain descriptions, model setup,
and the main evaluation loop for zero-shot PDDL plan verification.
"""

import glob
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import dotenv
import dspy
dotenv.load_dotenv()
from baselines.action_executor import (
    ActionExecutor,
    get_executor,
    parse_actions_syntax,
    parse_facts_syntax,
)
import re

# ============================================================================
# PDDL PARSING UTILITIES
# ============================================================================

class PDDLAction:
    def __init__(self, name, parameters, preconditions, effects):
        self.name = name
        self.parameters = parameters  # List of var names e.g. ['?x']
        self.preconditions = preconditions # List of (is_pos, predicate, args)
        self.effects = effects # List of (is_pos, predicate, args)

    def __repr__(self):
        return f"Action({self.name}, params={self.parameters})"

class PDDLParser:
    """A simple regex-based parser for STRIPS PDDL actions."""
    def __init__(self, pddl_str: str):
        self.pddl_str = pddl_str
        self.actions = {}
        self.parse_actions()

    def parse_actions(self):
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', self.pddl_str)
        # Find all (:action ... ) blocks
        # This regex is a bit simplistic, assuming standard formatting
        action_pattern = re.compile(r'\(:action\s+(\S+)\s+:parameters\s*\(([^)]*)\)\s+:precondition\s*(.*?)\s+:effect\s*(.*?)\s*\)', re.IGNORECASE)
        
        # We need a better way to capture nested parens for precondition/effect
        # Let's simple scan the string
        idx = 0
        while True:
            idx = text.find('(:action', idx)
            if idx == -1: break
            
            # Extract the full action block by balancing parens
            end_idx = self._find_closing_paren(text, idx)
            if end_idx == -1: break
            
            action_block = text[idx:end_idx+1]
            self._parse_single_action(action_block)
            idx = end_idx + 1

    def _find_closing_paren(self, text, start_idx):
        count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '(': count += 1
            elif text[i] == ')': count -= 1
            if count == 0: return i
        return -1

    def _parse_single_action(self, block):
        # Extract name
        name_match = re.search(r'\(:action\s+(\S+)', block, re.IGNORECASE)
        if not name_match: return
        name = name_match.group(1).lower()

        # Extract parameters
        params_match = re.search(r':parameters\s*\(([^)]*)\)', block, re.IGNORECASE)
        params = params_match.group(1).split() if params_match else []
        params = [p.strip() for p in params if p.strip().startswith('?')]

        self.actions[name] = PDDLAction(name, params, [], [])

def load_pddl_domain(domain_name: str) -> Optional[str]:
    """Finds a train file for the domain and extracts PDDL."""
    pattern = f"data/train/{domain_name}-training-dev_*.json"
    files = glob.glob(pattern)
    if not files:
        # Fallback for domains like 'grippers' where file might be named differently?
        # Standard naming seems consistent: {domain}-training-dev...
        return None
    
    # Read first file
    try:
        with open(files[0], 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                # Some datasets might have it in the first item
                return data[0].get("PDDL_domain")
    except Exception as e:
        print(f"Error loading PDDL for {domain_name}: {e}")
        return None
    return None
# ============================================================================
# DOMAIN DESCRIPTIONS
# ============================================================================

DOMAIN_DESCRIPTIONS = {
        "blocksworld": """This is a blocksworld domain where blocks can be placed on top of each other or on the table. 
        There is one robotic arm that can manipulate blocks. The arm can pick up a block (if it's clear and 
        the arm is empty), put down a block (if the arm is holding it), stack a block on another block 
        (if the target block is clear and the arm is holding a block), and unstack a block from another 
        block (if the block is clear and the arm is empty). Key constraints: only one block can be on top 
        of another, the arm can hold at most one block, and only clear blocks (with nothing on top) can be moved.
        The 'clear' predicate indicates a block has nothing on top of it, 'on' indicates one block is directly 
        on another, 'ontable' means a block is directly on the table, 'holding' means the arm is holding a block, 
        and 'handempty' means the arm is not holding anything.""",
        
        "ferry": """This is a ferry transportation domain where a ferry moves cars between different locations. 
        The ferry can sail between locations, and cars can board (embark) and disembark from the ferry. 
        Key constraints: the ferry can only be at one location at a time, cars can only board the ferry 
        when both the car and ferry are at the same location, and cars can only disembark when the ferry 
        is at their destination. The 'at' predicate indicates a car's location, 'at-ferry' shows the ferry's 
        current location, 'on' means a car is on the ferry, 'empty-ferry' indicates the ferry has no cars, 
        and 'not-eq' establishes that locations are distinct from each other.""",
        
        "logistics": """This is a logistics domain involving packages, trucks, airplanes, and locations. 
        Packages need to be transported between cities using trucks (for local transport within a city) 
        and airplanes (for transport between cities). Key constraints: trucks can only operate within 
        their home city, airplanes can fly between airports, packages must be loaded onto vehicles 
        before transport, and vehicles must be at the same location as packages to load them. 
        The 'at' predicate shows locations of objects, 'in' indicates a package is in a vehicle, 
        'in-city' establishes which city a location belongs to.""",
        
        "grippers": """This is a grippers domain where robots with grippers move balls between rooms. 
        Each robot has two grippers and can carry at most two balls simultaneously. Robots can move 
        between rooms, pick up balls (if they have a free gripper and are in the same room), and 
        drop balls in rooms. Key constraints: each gripper can hold at most one ball, robots must 
        be in the same room as a ball to pick it up, and balls can only be in one location at a time. 
        The 'at' predicate indicates locations, 'carry' shows which gripper is holding which ball, 
        and 'free' indicates an available gripper.""",
        
        "rovers": """This is a planetary rover domain where rovers navigate terrain to collect samples and data. 
        Rovers can move between waypoints, take images with cameras, collect soil/rock samples, and transmit 
        data to landers. Key constraints: rovers have limited battery and storage, some waypoints may be 
        unreachable due to terrain, and certain instruments are required for specific objectives. 
        The 'at' predicate shows rover locations, 'have_soil_analysis' indicates collected samples, 
        'communicated_soil_data' shows transmitted information.""",
        
        "visitall": """This is a visit-all domain where an agent must visit every location exactly once. 
        The agent can move between connected locations but cannot revisit locations they've already been to. 
        Key constraints: each location can only be visited once, the agent can only move between directly 
        connected locations, and the goal is typically to visit all locations. The 'at' predicate shows 
        the agent's current location, 'visited' indicates which locations have been visited, and 'connected' 
        defines valid movement paths between locations.""",
        
        "grid": """This is a grid navigation domain where an agent moves on a rectangular grid to reach 
        target positions or collect objects. The agent can move up, down, left, or right to adjacent 
        grid cells. Key constraints: the agent cannot move outside the grid boundaries, some cells may 
        be blocked or contain obstacles, and movement is typically to adjacent cells only. The 'at' 
        predicate indicates the agent's position, 'adjacent' defines valid moves between grid cells.""",
        
        "floortile": """This is a floor tiling domain where robots paint tiles on a floor in specific colors 
        and patterns. Robots can move between adjacent tiles and paint tiles they are standing on. 
        Key constraints: robots can only paint the tile they are currently on, some tiles may already 
        be painted and cannot be changed, and robots must coordinate to avoid conflicts. The 'robot-at' 
        predicate shows robot locations, 'painted' indicates which tiles have been painted with which colors, 
        and 'adjacent' defines movement possibilities between tiles.""",

        "alfworld": """This is an embodied household-task domain adapted from ALFWorld/ALFRED, 
        where an agent interacts with everyday objects inside indoor environments such as kitchens, 
        bedrooms, and living rooms. The agent can navigate rooms, open and close containers (like drawers, 
        cabinets, fridges), pick up and put down objects, toggle appliances on or off, and place items in 
        target receptacles. Key constraints: the agent must be in the same room and within reach of an object 
        to manipulate it; only open containers can be accessed; and objects have unique types and allowed 
        receptacles. The 'at' predicate represents an agent’s or object’s location, 'in' indicates containment, 
        'open' shows whether a container is open, and 'holding' indicates the agent is carrying an object.""",

        "depot": """This is a supply-depot management domain where pallets, crates, and hoists interact 
        within warehouses. Crates are stored on pallets, hoists can lift and move crates, trucks arrive to 
        load crates, and pallets can stack crates in restricted configurations. Key constraints: a hoist can 
        carry at most one crate, crates can only be moved when fully supported, and trucks must be at a docking 
        bay to load or unload goods. The 'on' predicate indicates crate stacking, 'at' shows the locations of 
        trucks or hoists, 'in' indicates crates loaded onto trucks, and 'lifting' shows when a hoist is holding 
        a crate.""",

        "goldminer": """This is a mining-and-resource-collection domain where a miner navigates tunnels to 
        extract gold chunks and deliver them to a safe location. The agent can move between adjacent tunnel 
        cells, dig to uncover hidden gold, pick up gold pieces, and drop them at designated collection sites. 
        Key constraints: digging may be required before gold becomes accessible, the miner can carry only a 
        limited number of gold pieces at once, and some cells may be blocked. The 'at' predicate shows the 
        miner’s location, 'gold-at' indicates the location of gold pieces, and 'carrying' specifies which gold 
        items the miner currently holds.""",

        "satellite": """This is a satellite-imaging domain where satellites capture images of celestial 
        targets using specialized instruments. Satellites can turn to face different directions, calibrate 
        their instruments using calibration targets, and take images once calibration is valid. Key constraints: 
        instruments must be calibrated immediately before use, satellites can only point in one direction at a 
        time, and some satellites have restricted slewing capabilities. The 'pointing' predicate indicates the 
        satellite's current orientation, 'calibrated' shows whether an instrument is currently calibrated, and 
        'have-image' indicates that an image of a target has been successfully taken.""",

        "swap": """This is a swapping domain where agents exchange the positions of objects across a set of 
        locations. Objects occupy unique slots, and the agent can swap the contents of two slots if both are 
        accessible. Key constraints: each location can hold exactly one object; swaps are atomic (a pairwise 
        exchange); and some locations may not be directly swappable unless intermediate steps are taken. 
        The 'at' predicate shows which object is in which location, and 'swappable' indicates whether two 
        positions can be legally swapped in a single action."""
    }


def load_test_data(data_dir: str = "data/test", domains: Optional[List[str]] = None, ids: Optional[List[str]] = None) -> List[Dict]:
    """Load test data from JSON files matching pattern."""
    test_files = sorted(glob.glob(os.path.join(data_dir, "*-test-dev_*.json")))
    
    # Filter by specific domains if provided
    if domains:
        filtered_files = []
        for d in domains:
            for f in test_files:
                if d.lower() in os.path.basename(f).lower():
                    filtered_files.append(f)
        test_files = filtered_files
    
    all_examples = []
    print(f"Loading data from {len(test_files)} files...")
    
    for file_path in test_files:
        basename = os.path.basename(file_path)
        domain = basename.split('-')[0]
        
        try:
            with open(file_path, "r") as f:
                items = json.load(f)
                
            for item in items:
                # Filter by ID if specified
                if ids:
                    item_id = str(item.get("id", item.get("example_id", "")))
                    if item_id not in ids:
                        continue
                
                item["domain"] = domain
                all_examples.append(item)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return all_examples

# ============================================================================
# DSPY SIGNATURE AND MODULE
# ============================================================================

class ZeroShotSignature(dspy.Signature):
    """
    Given a domain description and problem context, determine the initial state, goal state,
    plan (sequence of actions), and whether the goal is reachable.

    Format states as lists of predicates: (predicate arg1 arg2 ...)
    Format actions as: (action-name arg1 arg2 ...)
    """
    domain_description = dspy.InputField(desc="Description of the domain mechanics.")
    available_actions = dspy.InputField(desc="List of available actions in the domain.")
    context = dspy.InputField(desc="The problem setup describing objects and relations.")
    question = dspy.InputField(desc="The question to answer about reachability.")

    reasoning = dspy.OutputField(desc="Step-by-step reasoning about the problem.")
    initial_state = dspy.OutputField(desc="The initial state as STRIPS predicates. No comments or explanations.")
    goal_state = dspy.OutputField(desc="The goal state as STRIPS predicates. No comments or explanations.")
    list_of_actions = dspy.OutputField(desc="Sequence of actions to reach goal in correct order. Write 'Not Applicable' if impossible.")
    final_answer = dspy.OutputField(desc="'yes' or 'no'.")


class ZeroShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(ZeroShotSignature)

    def forward(self, domain_desc: str, available_actions: str, context: str, question: str):
        return self.prog(
            domain_description=domain_desc,
            available_actions=available_actions,
            context=context,
            question=question
        )


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model(model_type: str = "openai", model_name: str = "gpt-5.1") -> bool:
    """Setup DSPy language model configuration
    
    Returns:
        bool: True if DSPy thinking should be enabled, False if disabled (for thinking models)
    """
    gemini_api_key = os.getenv('GEMINI_API_KEY', 'GEMINI_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY', 'OPENAI_API_KEY')
    
    if model_type.lower() == "openai":
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY', 'OPENAI_API_KEY')
        lm = dspy.LM(f'openai/{model_name}', api_key=openai_api_key, temperature=1.0,max_tokens=64000, reasoning_effort="medium",cache=False)
        print(f"✓ Configured GPT-5.1 model ({model_name})")
        has_built_in_thinking = True  # GPT-5.1 does have built-in thinking mode

    elif model_type.lower() == "gemini":
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'GEMINI_API_KEY')
        lm = dspy.LM(f'gemini/{model_name}', api_key=gemini_api_key, max_tokens=32000,cache=False)
        print(f"✓ Configured Gemini model ({model_name})")
        has_built_in_thinking = False
    elif model_type.lower() == "rits":
        rits_url = os.getenv('RITS_API_URL', None) 
        rits_key = os.getenv('RITS_API_KEY', None) 
    
        lm = dspy.LM(f"openai/{model_name}",api_key="NotRequired",api_base=f"{rits_url}/v1",headers={"RITS_API_KEY": rits_key}, max_tokens=64000)
        print(f"✓ Configured RITS model ({model_name})")
        has_built_in_thinking = False # Only for OSS  
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'gpt5.1' or 'gemini'")
    
    dspy.configure(lm=lm)
    return not has_built_in_thinking


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_plan(
    executor: ActionExecutor,
    initial_state: str,
    goal_state: str,
    actions_str: str,
    debug: bool = False
) -> Dict[str, Any]:
    """Verify a plan using the action executor."""
    # Parse states and actions
    init_facts = parse_facts_syntax(initial_state)
    goal_facts = parse_facts_syntax(goal_state)
    actions = parse_actions_syntax(actions_str)
    
    if debug:
        print(f"  Initial facts ({len(init_facts)}): {init_facts[:5]}...")
        print(f"  Goal facts ({len(goal_facts)}): {goal_facts[:5]}...")
        print(f"  Actions ({len(actions)}): {actions[:3]}...")
    
    # Initialize state
    executor.set_state(init_facts)
    
    # Check if actions are empty or "not applicable"
    if not actions or (len(actions) == 1 and "not" in actions[0][0].lower()):
        goal_satisfied = executor.check_goal(goal_facts)
        return {
            "valid": goal_satisfied,
            "reason": "Goal already satisfied" if goal_satisfied else "No actions and goal not satisfied",
            "actions_applied": 0,
            "total_actions": 0,
        }
    
    # Apply each action
    executed_final_state = ""
    error_msg = None
    actions_applied = 0
    
    success = True
    for i, (action_name, args) in enumerate(actions):
        if not executor.apply_action(action_name, args):
            success = False
            error_msg = f"Action {i+1} '{action_name}' failed: {executor.last_error}"
            actions_applied = i
            break
        actions_applied = i + 1
            
    # Capture executed state
    executed_state_list = sorted([f"({p} {' '.join(a)})" for p, a in executor.state])
    executed_final_state = " ".join(executed_state_list)
    
    # Check goal
    goal_satisfied = executor.check_goal(goal_facts)
    
    result = {
        "valid": goal_satisfied and success,
        "reason": error_msg if not success else ("Goal satisfied" if goal_satisfied else "Goal not satisfied after all actions"),
        "actions_applied": actions_applied,
        "total_actions": len(actions),
        "executed_final_state": executed_final_state
    }
    
    # Handle the "no actions" case (valid if goal already satisfied)
    if not actions or (len(actions) == 1 and "not" in actions[0][0].lower()):
         result["actions_applied"] = 0
         result["total_actions"] = 0
         result["valid"] = goal_satisfied
         result["reason"] = "Goal already satisfied" if goal_satisfied else "No actions and goal not satisfied"
         
    return result


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================




def get_available_actions(domain: str) -> str:
    """Get a description of available actions for a domain by parsing PDDL from training data."""
    pddl_str = load_pddl_domain(domain)
    if not pddl_str:
        return "Could not load PDDL actions."
    
    parser = PDDLParser(pddl_str)
    action_strs = []
    
    # Sort for deterministic output
    for name in sorted(parser.actions.keys()):
        action = parser.actions[name]
        params = ", ".join(action.parameters)
        action_strs.append(f"{name}({params})")
        
    return ", ".join(action_strs)


def run_evaluation(
    data_dir: str,
    output_path: str,
    domains: List[str] = None,
    ids: List[str] = None,
    limit: int = None,
    debug: bool = False,
):
    """
    Run evaluation on test data and save results in hierarchical format matching zero_shot.py.
    """
    # Load all JSON files from data directory
    files = sorted([f for f in glob.glob(os.path.join(data_dir, "*.json"))])
    if not files:
        print(f"No JSON files found in {data_dir}")
        return

    # Initialize structure
    all_results = {
        "summary": {},
        "domains": {}
    }
    
    # Save initialized empty structure immediately to overwrite old format
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Global accumulators for summary
    total_examples_all = 0
    total_correct_all = 0
    total_correct_with_plan_all = 0
    total_plan_verified_all = 0
    total_plan_yes_all = 0
    
    count = 0
    
    print(f"Starting evaluation on {len(files)} domain files...")
    
    for file_path in files:
        filename = os.path.basename(file_path).replace('.json', '')
        # Extract base domain name: e.g. "alfworld-test-dev..." -> "alfworld"
        # Handle "blocksworld-4ops" special case if needed, but mostly it's the prefix
        domain = filename.split('-')[0]
        if filename.startswith("blocksworld-4ops"):
             domain = "blocksworld-4ops"
        
        # Filter domains
        if domains and domain not in domains:
            continue
            
        print(f"\nProcessing domain: {domain} (File: {filename})")
        
        # Get available actions for prompt and metadata
        pddl_str = load_pddl_domain(domain)
        valid_actions_list = []
        if pddl_str:
            parser = PDDLParser(pddl_str)
            valid_actions_list = sorted(list(parser.actions.keys()))

        # Initialize domain stats
        if domain not in all_results["domains"]:
            all_results["domains"][domain] = {
                "total_examples": 0,
                "correct_answer": 0,
                "correct_with_plan": 0,
                "plan_verified": 0,
                "total_yes_examples": 0,
                "accuracy": 0.0,
                "plan_accuracy": 0.0,
                "pddl_domain": pddl_str if pddl_str else "N/A",
                "valid_actions": valid_actions_list,
                "details": []
            }
        
        domain_res = all_results["domains"][domain]

            
        # Reconstruct available_actions string for prompt from the list to avoid double parsing
        available_actions = get_available_actions(domain) # Keep original call or optimize? 
        # Actually, get_available_actions does formatting with params which we need for prompt.
        # But we can optimize if we wanted. For safety, let's just call existing function or reuse parser.
        # Let's reuse the parser logic to be efficient if pddl_str loaded
        if pddl_str:
             # Already parsed above
             action_strs = []
             for name in sorted(parser.actions.keys()):
                 action = parser.actions[name]
                 params = ", ".join(action.parameters)
                 action_strs.append(f"{name}({params})")
             available_actions = ", ".join(action_strs)
        else:
             available_actions = get_available_actions(domain) # Fallback

        # Assuming domain description is not needed or fixed for now based on previous code
        domain_desc_file = os.path.join(os.path.dirname(data_dir), "train", f"{domain}.json") # Attempt to find description from train?
        # Re-using simple domain description logic from original function if present
        # In original function it read data_dir/*.json directly.
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Iterate over examples
        for ex in data:
            example_id = ex.get("id", ex.get("example_id", f"example_{count}"))
            
            # Filter IDs
            if ids and str(example_id) not in ids:
                continue
                
            if limit and count >= limit:
                break
                
            count += 1
            print(f"Processing example {count} ({domain}/{example_id})...")
            
            context = ex.get("context", "")
            question = ex.get("question", ex.get("inputs", ""))
            # Ensure ground truth is "yes" or "no"
            ground_truth = str(ex.get("answer", "")).lower()
            
            # Create domain description for prompt
            domain_name_key = domain.replace("-4ops", "") # simple normalization if needed
            domain_desc_text = DOMAIN_DESCRIPTIONS.get(domain_name_key, DOMAIN_DESCRIPTIONS.get(domain, f"Domain: {domain}"))
            
            # The prompt expected domain_desc to describe the domain semantics
            domain_desc = domain_desc_text

            try:
                # Run DSPy module
                module = ZeroShotModule() # Initialize module here to ensure fresh state for each example if needed, or outside loop if shared
                response = module(
                    domain_desc=domain_desc,
                    available_actions=available_actions,
                    context=context,
                    question=question
                )
                
                predicted_answer = response.final_answer.strip().lower()
                initial_state = response.initial_state
                goal_state = response.goal_state
                actions = response.list_of_actions
                reasoning = response.reasoning
                
                if debug:
                    print(f"  Predicted: {predicted_answer}, Ground truth: {ground_truth}")
                
                # Verify plan if predicted yes
                verification = {"valid": None, "reason": "Not verified"}
                
                if "yes" in predicted_answer:
                    try:
                        executor = get_executor(domain)
                        verification = verify_plan(
                            executor, initial_state, goal_state, actions, debug=debug
                        )
                    except Exception as e:
                        verification = {"valid": False, "reason": f"Verification error: {str(e)}"}
                
                # Determine correctness
                is_correct = predicted_answer == ground_truth
                is_yes_gt = ground_truth == "yes"
                is_yes_pred = "yes" in predicted_answer
                
                plan_valid_val = verification.get("valid")
                # Map None to "N/A" for uniformity with zero_shot.py
                plan_valid_display = plan_valid_val if plan_valid_val is not None else "N/A"
                
                executed_final_state = verification.get("executed_final_state", "N/A")
                failure_reason = verification.get("reason")
                
                # Calculate metric flags
                is_correct_with_plan = False
                if is_yes_gt:
                    if is_yes_pred and plan_valid_val is True:
                        is_correct_with_plan = True
                else: # gt is no
                    if not is_yes_pred: # correct prediction of no
                        is_correct_with_plan = True
                
                # Update Domain Stats
                domain_res["total_examples"] += 1
                if is_correct:
                    domain_res["correct_answer"] += 1
                if is_correct_with_plan:
                    domain_res["correct_with_plan"] += 1
                
                if is_yes_gt:
                    domain_res["total_yes_examples"] += 1
                    if plan_valid_val is True:
                        domain_res["plan_verified"] += 1
                
                # Recalculate accuracies
                if domain_res["total_examples"] > 0:
                    domain_res["accuracy"] = domain_res["correct_answer"] / domain_res["total_examples"]
                if domain_res["total_yes_examples"] > 0:
                    domain_res["plan_accuracy"] = domain_res["plan_verified"] / domain_res["total_yes_examples"]
                
                # Construct result item
                item = {
                    "id": example_id,
                    "context": context,
                    "question": question,
                    "gt_answer": ground_truth,
                    "pred_answer": predicted_answer,
                    "correct": is_correct,
                    "reasoning": reasoning,
                    "plan_valid": plan_valid_display,
                    "generated_plan": actions,
                    "generated_initial_state": initial_state,
                    "generated_goal_state": goal_state,
                    "executed_final_state": executed_final_state,
                    "plan_failure_reason": failure_reason,
                    "verification": verification
                }
                
                domain_res["details"].append(item)
                
                # Update Global Stats
                total_examples_all += 1
                if is_correct: total_correct_all += 1
                if is_correct_with_plan: total_correct_with_plan_all += 1
                if is_yes_gt:
                    total_plan_yes_all += 1
                    if plan_valid_val is True:
                        total_plan_verified_all += 1
                
                # Update Summary
                all_results["summary"] = {
                     "overall_accuracy": total_correct_all / total_examples_all if total_examples_all else 0,
                     "overall_accuracy_with_plan": total_correct_with_plan_all / total_examples_all if total_examples_all else 0,
                     "overall_plan_accuracy": total_plan_verified_all / total_plan_yes_all if total_plan_yes_all else 0,
                     "total_examples": total_examples_all,
                     "total_correct": total_correct_all,
                     "total_correct_with_plan": total_correct_with_plan_all,
                     "total_plan_verified": total_plan_verified_all,
                     "total_plan_possible": total_plan_yes_all
                }
                
                # Save incrementally
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                
            except Exception as e:
                print(f"  Error: {e}")
                # Log error item?
                # Matching zero_shot.py might simply skip or partial log
                pass
            
        if limit and count >= limit:
            break
    
    # Final print
    print(f"\nEvaluation Complete.")
    print(json.dumps(all_results["summary"], indent=2))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Zero-shot PDDL plan verification with DSPy")
    parser.add_argument("--data", type=str, default="data/test_baseline", help="Path to test data directory")
    parser.add_argument("--output", type=str, default="baselines/zero_shot_results.json", help="Output path")
    parser.add_argument("--domains", type=str, nargs="+", help="Filter by domains")
    parser.add_argument("--id", type=str, nargs="+", help="Filter by example IDs")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--model-type", type=str, default="openai", choices=["openai", "gemini", "rits"])
    parser.add_argument("--model-name", type=str, default="gpt-5.1")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Setup model
    setup_model(args.model_type, args.model_name)
    
    # Run evaluation
    run_evaluation(
        data_dir=args.data,
        output_path=args.output,
        domains=args.domains,
        ids=args.id,
        limit=args.limit,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
