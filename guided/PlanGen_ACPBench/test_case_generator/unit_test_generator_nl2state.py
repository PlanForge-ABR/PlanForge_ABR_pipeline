#!/usr/bin/env python3
"""
Unit Test Generator for NL2State
Generates test cases from training data for natural language to state conversion tasks.

Usage:
    python unit_test_generator_nl2state.py --train <train_dir> --N <num_cases> --domain <domain_name> --out <output_file> [--seed <random_seed>]
    --train: Directory containing training data JSON files.
    --N: Number of test cases to generate.
    --domain: Domain name (e.g., ferry, blocksworld, logistics, grippers, rovers, visitall, grid, floortile).
    --out: Output JSON file path.
    --seed: Random seed for reproducibility (default: 42).
Example:
    python test_case_generator/unit_test_generator_nl2state.py --train ./data/train --N 10 --domain blocksworld --out nl2state_test.json
"""

import json
import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import unified parser
from unified_pddl_parser import unified_parser

try:
    from tarski.io import PDDLReader
    from tarski.syntax import Atom, CompoundFormula
    from tarski.syntax.formulas import land
    TARSKI_AVAILABLE = True
except ImportError:
    print("Warning: tarski library not found, using fallback parsing")
    PDDLReader = None
    Atom = None
    CompoundFormula = None
    land = None
    TARSKI_AVAILABLE = False


class PDDL2StateConverter:
    """Converts PDDL problem definitions to state representations using tarski.io"""
    
    _tarski_failures = 0
    _use_tarski = True
    
    @staticmethod
    def parse_pddl_state(pddl_domain: str, pddl_problem: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse PDDL domain and problem to extract initial state and goal state using unified parser
        
        Args:
            pddl_domain: PDDL domain definition string
            pddl_problem: PDDL problem definition string
            
        Returns:
            Tuple of (initial_state_predicates, goal_state_predicates)
        """
        try:
            # Use unified parser
            parsed_results = unified_parser.parse_pddl_pair(pddl_domain, pddl_problem, validate=TARSKI_AVAILABLE)
            
            # Extract initial state predicates
            initial_state_predicates = parsed_results["initial_state"]
            initial_state = [{"predicate": pred, "args": args} for pred, args in initial_state_predicates]
            
            # Extract goal state predicates from problem tree
            problem_tree = parsed_results["problem_tree"]
            goal_state = PDDL2StateConverter._extract_goal_from_tree(problem_tree)
            
            return initial_state, goal_state
            
        except Exception as e:
            print(f"Warning: Unified parser failed, using improved fallback: {e}")
            PDDL2StateConverter._tarski_failures += 1
            if PDDL2StateConverter._tarski_failures > 5:
                PDDL2StateConverter._use_tarski = False
                print("Too many parsing failures, disabling unified parsing")
            return PDDL2StateConverter.improved_fallback_parse_pddl_state(pddl_problem)

    @staticmethod  
    def _extract_goal_from_tree(problem_tree) -> List[Dict]:
        """Extract goal predicates from parsed problem tree"""
        def walk_tree(node):
            goals = []
            if isinstance(node, list):
                if len(node) > 0 and node[0] == ":goal":
                    # Found goal section
                    for item in node[1:]:
                        if isinstance(item, list):
                            goals.extend(extract_predicates(item))
                else:
                    # Recursively search sublists
                    for item in node:
                        goals.extend(walk_tree(item))
            return goals
        
        def extract_predicates(goal_expr):
            """Extract predicates from goal expression"""
            preds = []
            if isinstance(goal_expr, list):
                if len(goal_expr) > 0:
                    if goal_expr[0] == "and":
                        # Conjunction of predicates
                        for pred in goal_expr[1:]:
                            if isinstance(pred, list) and pred:
                                pred_name = str(pred[0])
                                args = [str(arg) for arg in pred[1:]]
                                preds.append({"predicate": pred_name, "args": args})
                    else:
                        # Single predicate
                        pred_name = str(goal_expr[0])
                        args = [str(arg) for arg in goal_expr[1:]]
                        preds.append({"predicate": pred_name, "args": args})
            return preds
        
        return walk_tree(problem_tree)

    @staticmethod
    def _extract_state_predicates(state) -> List[Dict]:
        """Extract predicates from tarski state representation"""
        predicates = []
        
        if hasattr(state, 'as_atoms'):
            # Get all atoms in the state
            for atom in state.as_atoms():
                predicate_dict = PDDL2StateConverter._atom_to_dict(atom)
                if predicate_dict:
                    predicates.append(predicate_dict)
        else:
            # If state is iterable of atoms
            try:
                for atom in state:
                    predicate_dict = PDDL2StateConverter._atom_to_dict(atom)
                    if predicate_dict:
                        predicates.append(predicate_dict)
            except (TypeError, AttributeError):
                print(f"Warning: Could not extract predicates from state type: {type(state)}")
        
        return predicates
    
    @staticmethod
    def _extract_goal_predicates(goal) -> List[Dict]:
        """Extract predicates from tarski goal representation"""
        predicates = []
        
        def extract_from_formula(formula):
            if isinstance(formula, Atom):
                predicate_dict = PDDL2StateConverter._atom_to_dict(formula)
                if predicate_dict:
                    predicates.append(predicate_dict)
            elif isinstance(formula, CompoundFormula):
                # Handle compound formulas (AND, OR, etc.)
                if hasattr(formula, 'subformulas'):
                    for subformula in formula.subformulas:
                        extract_from_formula(subformula)
                elif hasattr(formula, 'children'):
                    for child in formula.children:
                        extract_from_formula(child)
        
        try:
            extract_from_formula(goal)
        except Exception as e:
            print(f"Warning: Error extracting goal predicates: {e}")
        
        return predicates
    
    @staticmethod
    def _atom_to_dict(atom) -> Optional[Dict]:
        """Convert tarski Atom to dictionary format"""
        try:
            predicate_name = str(atom.predicate)
            # Remove arity suffix (e.g., "on/2" -> "on", "at/1" -> "at")
            if '/' in predicate_name:
                predicate_name = predicate_name.split('/')[0]
            
            args = [str(arg) for arg in atom.subterms] if hasattr(atom, 'subterms') else []
            
            return {
                "predicate": predicate_name,
                "args": args
            }
        except Exception as e:
            print(f"Warning: Could not convert atom {atom} to dict: {e}")
            return None
    
    @staticmethod
    def _improved_fallback_parse_pddl_state(pddl_problem: str) -> Tuple[List[Dict], List[Dict]]:
        """Improved fallback regex-based parsing when tarski fails"""
        import re
        
        initial_state = []
        goal_state = []
        
        # Clean up the PDDL problem string
        pddl_problem = pddl_problem.replace('\n', ' ').replace('\t', ' ')
        # Normalize whitespace
        pddl_problem = re.sub(r'\s+', ' ', pddl_problem)
        
        # Extract initial state - look for :init section
        init_pattern = r':init\s*\((.*?)\)\s*\(:goal'
        init_match = re.search(init_pattern, pddl_problem, re.DOTALL)
        if init_match:
            init_content = init_match.group(1).strip()
            initial_state = PDDL2StateConverter._parse_predicates_improved(init_content)
        
        # Extract goal state - handle both (and ...) and direct goal formats
        goal_pattern1 = r':goal\s*\(and\s+(.*?)\)\s*\)'
        goal_pattern2 = r':goal\s*\(([^)]+(?:\([^)]*\)[^)]*)*)\)\s*\)'
        
        goal_match = re.search(goal_pattern1, pddl_problem, re.DOTALL)
        if goal_match:
            goal_content = goal_match.group(1).strip()
            goal_state = PDDL2StateConverter._parse_predicates_improved(goal_content)
        else:
            goal_match = re.search(goal_pattern2, pddl_problem, re.DOTALL)
            if goal_match:
                goal_content = goal_match.group(1).strip()
                goal_state = PDDL2StateConverter._parse_predicates_improved(goal_content)
            
        return initial_state, goal_state
    
    @staticmethod
    def _parse_predicates_improved(content: str) -> List[Dict]:
        """Improved predicate parsing using regex"""
        import re
        predicates = []
        
        if not content:
            return predicates
        
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Find all predicate expressions using a more robust pattern
        # This pattern looks for opening parenthesis, captures content until matching closing parenthesis
        predicate_pattern = r'\(([^()]+(?:\s+[^()]+)*)\)'
        matches = re.findall(predicate_pattern, content)
        
        for match in matches:
            parts = match.strip().split()
            if parts:
                predicate_name = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                predicates.append({
                    "predicate": predicate_name,
                    "args": args
                })
        
        # If the simple pattern doesn't work, try the balanced parentheses approach
        if not predicates and content:
            predicates = PDDL2StateConverter._parse_predicates_regex(content)
        
        return predicates
    
    @staticmethod
    def _fallback_parse_pddl_state(pddl_problem: str) -> Tuple[List[Dict], List[Dict]]:
        """Original fallback regex-based parsing when tarski fails"""
        import re
        
        initial_state = []
        goal_state = []
        
        # Extract initial state
        init_match = re.search(r':init\s*\((.*?)\)\s*\(:goal', pddl_problem, re.DOTALL)
        if init_match:
            init_content = init_match.group(1)
            initial_state = PDDL2StateConverter._parse_predicates_regex(init_content)
        
        # Extract goal state
        goal_match = re.search(r':goal\s*\(and\s*(.*?)\)\s*\)', pddl_problem, re.DOTALL)
        if not goal_match:
            # Try without 'and' wrapper
            goal_match = re.search(r':goal\s*\((.*?)\)\s*\)', pddl_problem, re.DOTALL)
        
        if goal_match:
            goal_content = goal_match.group(1)
            goal_state = PDDL2StateConverter._parse_predicates_regex(goal_content)
            
        return initial_state, goal_state
    
    @staticmethod
    def _parse_predicates_regex(content: str) -> List[Dict]:
        """Parse predicate expressions using regex (fallback method)"""
        import re
        predicates = []
        
        # Find all predicate expressions (balanced parentheses)
        i = 0
        while i < len(content):
            if content[i] == '(':
                # Find matching closing parenthesis
                paren_count = 1
                start = i + 1
                j = i + 1
                
                while j < len(content) and paren_count > 0:
                    if content[j] == '(':
                        paren_count += 1
                    elif content[j] == ')':
                        paren_count -= 1
                    j += 1
                
                if paren_count == 0:
                    predicate_str = content[start:j-1].strip()
                    if predicate_str:
                        predicate = PDDL2StateConverter._parse_single_predicate_regex(predicate_str)
                        if predicate:
                            predicates.append(predicate)
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        return predicates
    
    @staticmethod
    def _parse_single_predicate_regex(predicate_str: str) -> Optional[Dict]:
        """Parse a single predicate string into structured format (fallback method)"""
        # Clean up the string
        predicate_str = predicate_str.strip()
        if not predicate_str:
            return None
        
        # Split by whitespace
        parts = predicate_str.split()
        if not parts:
            return None
        
        predicate_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        return {
            "predicate": predicate_name,
            "args": args
        }


class DomainDescriptionGenerator:
    """Generates detailed domain descriptions based on domain type and PDDL content"""
    
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
        and 'adjacent' defines movement possibilities between tiles."""
    }
    
    @classmethod
    def get_domain_description(cls, domain: str, pddl_domain: str = "") -> str:
        """Get detailed domain description"""
        domain_lower = domain.lower()
        
        if domain_lower in cls.DOMAIN_DESCRIPTIONS:
            return cls.DOMAIN_DESCRIPTIONS[domain_lower]
        
        # Fallback: try to infer from PDDL domain if available
        if pddl_domain:
            return cls._infer_from_pddl(pddl_domain)
        
        return f"This is a {domain} planning domain with various objects and actions."
    
    @classmethod
    def _infer_from_pddl(cls, pddl_domain: str) -> str:
        """Attempt to infer domain description from PDDL domain definition"""
        import re
        # Extract predicates and actions for basic description
        predicates = re.findall(r'\(:predicates.*?\)', pddl_domain, re.DOTALL)
        actions = re.findall(r'\(:action\s+(\w+)', pddl_domain)
        
        description = "This is a planning domain"
        if actions:
            description += f" with actions including: {', '.join(actions[:5])}"
        if len(actions) > 5:
            description += " and others"
        
        return description + ". The domain involves coordinated planning to achieve specific goal states."


class NL2StateTestGenerator:
    """Main test case generator for NL2State"""
    
    def __init__(self):
        self.converter = PDDL2StateConverter()
        self.description_generator = DomainDescriptionGenerator()
    
    def load_training_data(self, train_dir: str, domain: str) -> List[Dict]:
        """Load training data from JSON files"""
        train_path = Path(train_dir)
        
        # Look for domain-specific training file
        domain_file = train_path / f"{domain}-training-dev_08_22_2026.json"
        if not domain_file.exists():
            # Try alternative naming patterns
            possible_files = list(train_path.glob(f"*{domain}*.json"))
            if not possible_files:
                raise FileNotFoundError(f"No training file found for domain '{domain}' in {train_dir}")
            domain_file = possible_files[0]
        
        print(f"Loading training data from: {domain_file}")
        
        with open(domain_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def extract_context_and_question(self, item: Dict) -> Tuple[str, str]:
        """Extract context and question from training item"""
        context = item.get('context', '')
        inputs = item.get('inputs', '')
        targets = item.get('targets', '')
        
        # The question is typically in the inputs field
        question = inputs
        
        # If inputs contains both context and question, try to separate them
        if 'context' in item and item['context']:
            full_context = context
        else:
            # Sometimes context is embedded in inputs
            if '\\n' in inputs and 'domain' in inputs.lower():
                parts = inputs.split('\\n')
                context_parts = [p for p in parts if 'domain' in p.lower() or 'block' in p.lower()]
                if context_parts:
                    full_context = ' '.join(context_parts)
                    # Remove context from question
                    for part in context_parts:
                        inputs = inputs.replace(part, '').replace('\\n', ' ').strip()
                    question = inputs
                else:
                    full_context = context
            else:
                full_context = context
        
        return full_context, question
    
    def convert_to_test_format(self, item: Dict, domain: str) -> Dict:
        """Convert training item to test case format"""
        # Extract domain description
        domain_description = self.description_generator.get_domain_description(
            domain, item.get('PDDL_domain', '')
        )
        
        # Extract context and question
        context, question = self.extract_context_and_question(item)
        
        # Parse PDDL problem to get states
        pddl_domain = item.get('PDDL_domain', '')
        pddl_problem = item.get('PDDL_problem', '')
        
        if not pddl_problem:
            print(f"Warning: No PDDL_problem found for item {item.get('id', 'unknown')}")
            return None
        
        if not pddl_domain:
            print(f"Warning: No PDDL_domain found for item {item.get('id', 'unknown')}")
            return None
        
        try:
            initial_predicates, goal_predicates = self.converter.parse_pddl_state(pddl_domain, pddl_problem)
            
            # Create output in required format
            output = {
                "initial_state": {
                    "state": initial_predicates
                },
                "goal_state": {
                    "state": goal_predicates
                }
            }
            
            # Create full context + question
            full_context_question = f"{context}\n{question}".strip()
            
            test_case = {
                "example_id": item.get('id', f"{domain}_example_{random.randint(1000, 9999)}"),
                "domain_description": domain_description,
                "context_and_question": full_context_question,
                "output": output
            }
            
            return test_case
            
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            return None
    
    def generate_test_cases(self, train_dir: str, domain: str, n_cases: int) -> List[Dict]:
        """Generate N test cases for the specified domain"""
        # Load training data
        training_data = self.load_training_data(train_dir, domain)
        
        if not training_data:
            raise ValueError(f"No training data found for domain {domain}")
        
        # Randomly sample N cases
        if n_cases > len(training_data):
            print(f"Warning: Requested {n_cases} cases but only {len(training_data)} available. Using all available cases.")
            selected_items = training_data
        else:
            selected_items = random.sample(training_data, n_cases)
        
        # Convert to test format
        test_cases = []
        for item in selected_items:
            test_case = self.convert_to_test_format(item, domain)
            if test_case:
                test_cases.append(test_case)
        
        return test_cases
    
    def save_test_cases(self, test_cases: List[Dict], output_file: str, domain: str):
        """Save test cases to JSON file"""
        output_data = {
            domain: test_cases
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Generated {len(test_cases)} test cases for domain '{domain}' and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate unit tests for NL2State")
    parser.add_argument('--train', required=True, help='Path to training data directory')
    parser.add_argument('--N', type=int, required=True, help='Number of test cases to generate')
    parser.add_argument('--domain', required=True, help='Domain name (e.g., ferry, blocksworld)')
    parser.add_argument('--out', required=True, help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create generator
    generator = NL2StateTestGenerator()
    
    try:
        # Generate test cases
        test_cases = generator.generate_test_cases(args.train, args.domain, args.N)
        
        # Save to output file
        generator.save_test_cases(test_cases, args.out, args.domain)
        
    except Exception as e:
        print(f"Error generating test cases: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())