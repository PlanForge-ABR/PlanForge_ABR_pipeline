import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional
from evaluation.action_executor import (
    ActionExecutor,
    get_executor,
    parse_facts_syntax,
    parse_actions_syntax,
)

def verify_plan(
    executor: ActionExecutor,
    initial_state_str: str,
    goal_state_str: str,
    plan: List[Any],
    debug: bool = False
) -> Dict[str, Any]:
    """Verify a plan using the action executor.
    
    Args:
        executor: The domain executor
        initial_state_str: Initial state string (JSON or PDDL-like)
        goal_state_str: Goal state string
        plan: List of actions. Can be strings "action(args)" or dictionaries/objects.
    """
    # Parse states
    # Note: initial_state_str in the JSON might be a JSON string representation of a list of dicts 
    # (e.g. "[{\"predicate\": ...}]").
    # parse_facts_syntax expects PDDL-like string or we might need to parse JSON.
    
    # Check if state strings are JSON lists of dicts
    if initial_state_str.strip().startswith("[") and "predicate" in initial_state_str:
        # It's likely the JSON format from ATLAS
        try:
            init_facts_raw = json.loads(initial_state_str)
            init_facts = []
            for f in init_facts_raw:
                # Convert {"predicate": "p", "args": ["a"]} to ("p", ["a"])
                init_facts.append((f["predicate"], f["args"]))
        except json.JSONDecodeError:
            # Fallback to PDDL parsing
            init_facts = parse_facts_syntax(initial_state_str)
    else:
        init_facts = parse_facts_syntax(initial_state_str)

    if goal_state_str.strip().startswith("[") and "predicate" in goal_state_str:
        try:
            goal_facts_raw = json.loads(goal_state_str)
            goal_facts = []
            for f in goal_facts_raw:
                goal_facts.append((f["predicate"], f["args"]))
        except json.JSONDecodeError:
            goal_facts = parse_facts_syntax(goal_state_str)
    else:
        goal_facts = parse_facts_syntax(goal_state_str)
        
    # Parse actions
    # The plan from ATLAS might be a list of strings or list of dicts?
    # Based on "generated_plan" in zero_shot, it's a list.
    # If the input `plan` is a list, we iterate.
    parsed_actions = []
    
    if not isinstance(plan, list):
         # If it's a string, try to parse it
         if isinstance(plan, str):
             parsed_actions = parse_actions_syntax(plan)
         else:
             parsed_actions = []
    else:
        for action_item in plan:
            if isinstance(action_item, str):
                # "action(arg)"
                parsed = executor.try_parse_action(action_item)
                if parsed:
                    parsed_actions.append(parsed)
                else:
                    # Fallback manual parse if simple string?
                    # try_parse_action should handle it.
                    pass
            elif isinstance(action_item, dict):
                # Handle dict format if ATLAS uses it: {"name": "...", "args": [...]}
                # I don't see this in the sample, but good to be robust
                name = action_item.get("name") or action_item.get("action")
                args = action_item.get("args") or action_item.get("parameters")
                if name:
                    parsed_actions.append((name, args or []))
            elif isinstance(action_item, list) or isinstance(action_item, tuple):
                 if len(action_item) >= 1:
                     parsed_actions.append((action_item[0], action_item[1] if len(action_item)>1 else []))

    if debug:
        print(f"  Initial facts ({len(init_facts)})")
        print(f"  Goal facts ({len(goal_facts)})")
        print(f"  Actions ({len(parsed_actions)})")

    # Initialize state
    executor.set_state(init_facts)

    # Check validity
    if not parsed_actions:
        goal_satisfied = executor.check_goal(goal_facts)
        return {
            "valid": goal_satisfied,
            "reason": "Goal already satisfied" if goal_satisfied else "No actions and goal not satisfied",
            "actions_applied": 0,
            "total_actions": 0
        }

    # Apply actions
    executed_final_state = ""
    error_msg = None
    actions_applied = 0
    success = True
    
    for i, (action_name, args) in enumerate(parsed_actions):
        if not executor.apply_action(action_name, args):
            success = False
            error_msg = f"Action {i+1} '{action_name}' failed: {executor.last_error}"
            actions_applied = i
            break
        actions_applied = i + 1
        
    # Check goal
    goal_satisfied = executor.check_goal(goal_facts)
    
    return {
        "valid": goal_satisfied and success,
        "reason": error_msg if not success else ("Goal satisfied" if goal_satisfied else "Goal not satisfied"),
        "actions_applied": actions_applied,
        "total_actions": len(parsed_actions)
    }

def main():
    src_dir = "src"
    pattern = os.path.join(src_dir, "*", "search_result_test_domain.json")
    files = glob.glob(pattern)
    
    if not files:
        print("No search_result_test_domain.json files found.")
        return

    all_results = {
        "summary": {},
        "domains": {}
    }

    # Global counters
    total_examples = 0
    total_correct = 0
    total_correct_with_plan = 0
    total_plan_verified = 0
    total_plan_possible = 0 # Total YES GT

    print(f"Found {len(files)} result files. Starting evaluation...")

    for file_path in sorted(files):
        domain = os.path.basename(os.path.dirname(file_path)) # src/<domain>/file.json
        print(f"\nProcessing domain: {domain}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
            
        domain_stats = {
            "total_examples": 0,
            "correct_answer": 0,
            "correct_with_plan": 0,
            "plan_verified": 0,
            "total_yes_examples": 0,
            "examples": []
        }
        
        executor = None
        try:
             executor = get_executor(domain)
        except Exception as e:
             print(f"  Warning: No executor for {domain}: {e}")
        
        for ex in data:
            domain_stats["total_examples"] += 1
            total_examples += 1
            
            # Extract fields
            gt_answer = str(ex.get("ground_truth_answer", "no")).lower()
            pred_answer = str(ex.get("final_answer", "no")).lower()
            plan = ex.get("plan", [])
            init_state = ex.get("predicted_initial_state", "")
            goal_state = ex.get("predicted_goal_state", "")
            
            is_correct = (gt_answer == pred_answer)
            if is_correct:
                domain_stats["correct_answer"] += 1
                total_correct += 1
                
            is_gt_yes = (gt_answer == "yes")
            is_pred_yes = (pred_answer == "yes")
            
            verification = {"valid": None, "reason": "N/A"}
            plan_valid = False
            
            if is_pred_yes and executor:
                # Assuming plan key contains the list of actions
                verify_res = verify_plan(executor, str(init_state), str(goal_state), plan)
                verification = verify_res
                if verify_res["valid"]:
                    plan_valid = True

            if is_gt_yes:
                domain_stats["total_yes_examples"] += 1
                total_plan_possible += 1
                if plan_valid:
                    domain_stats["plan_verified"] += 1
                    total_plan_verified += 1
            
            # Calculate correct_with_plan
            # If GT=YES, Pred=YES, Plan=Valid -> Correct with plan
            # If GT=NO, Pred=NO -> Correct with plan (trivial plan)
            is_correct_with_plan = False
            if is_gt_yes:
                if is_pred_yes and plan_valid:
                    is_correct_with_plan = True
            else:
                 # GT is NO
                 if not is_pred_yes: # Correctly predicted NO
                      is_correct_with_plan = True
                      
            if is_correct_with_plan:
                domain_stats["correct_with_plan"] += 1
                total_correct_with_plan += 1

            # Save detailed result
            domain_stats["examples"].append({
                "id": ex.get("example_id"),
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
                "plan_valid": verification.get("valid"),
                "plan_reason": verification.get("reason"),
                "correct_with_plan": is_correct_with_plan
            })
            
        all_results["domains"][domain] = domain_stats
        
        # Calculate domain accuracy
        acc = domain_stats["correct_answer"] / domain_stats["total_examples"] if domain_stats["total_examples"] else 0
        plan_acc = domain_stats["plan_verified"] / domain_stats["total_yes_examples"] if domain_stats["total_yes_examples"] else 0
        print(f"  Examples: {domain_stats['total_examples']}")
        print(f"  Accuracy: {acc:.2f}")
        print(f"  Plan Accuracy (on YES): {plan_acc:.2f}")

    # Summary
    overall_accuracy = total_correct / total_examples if total_examples else 0
    overall_accuracy_with_plan = total_correct_with_plan / total_examples if total_examples else 0
    overall_plan_accuracy = total_plan_verified / total_plan_possible if total_plan_possible else 0
    
    summary = {
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_with_plan": overall_accuracy_with_plan,
        "overall_plan_accuracy": overall_plan_accuracy,
        "total_examples": total_examples,
        "total_correct": total_correct,
        "total_correct_with_plan": total_correct_with_plan,
        "total_plan_verified": total_plan_verified,
        "total_plan_possible": total_plan_possible
    }
    
    all_results["summary"] = summary
    
    print("\nOverall Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save results
    with open("evaluation/atlas_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
if __name__ == "__main__":
    main()
