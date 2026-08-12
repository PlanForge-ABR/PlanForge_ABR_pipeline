
import sys
import os
import json
sys.path.append(os.path.join(os.getcwd(), "test_case_generator"))
from unit_test_generator_succ import SuccessorUnitTestGenerator
from unified_pddl_parser import unified_parser

# The example ID the user is concerned about
TARGET_ID = "frogs_jumping_frogs-3v3_p01_0_n"

def debug_specific_example():
    with open('data/train/frogs_jumping-training.json', 'r') as f:
        data = json.load(f)
        
    target_ex = None
    for ex in data:
        if ex.get('id') == TARGET_ID:
            target_ex = ex
            break
            
    if not target_ex:
        print(f"Example {TARGET_ID} not found!")
        return

    print(f"Found example {TARGET_ID}")
    domain_pddl = target_ex['PDDL_domain']
    problem_pddl = target_ex['PDDL_problem']
    
    # Instantiate generator just to access methods (or use static ones if possible)
    # We can use the logic directly from SuccessorUnitTestGenerator
    
    print("Parsing and generating successors...")
    try:
        # We'll use the logic embedded in compute_pairs_for_domain effectively
        # But we need to call the successor generation method directly.
        # SuccessorUnitTestGenerator has an instance method 'successors_from_init_fallback'
        # dependent on self.parser? No, it initializes self.parser.
        
        gen = SuccessorUnitTestGenerator(None, 1, None, None)
        
        # Manually invoke the successor logic
        # We need to see if it uses tarski or fallback
        if gen.TARKSI_AVAILABLE:
            print("Using Tarski path")
            dom_name, init_state, succ = gen.successors_from_init_tarski(domain_pddl, problem_pddl)
        else:
            print("Using Fallback path")
            dom_name, init_state, succ = gen.successors_from_init_fallback(domain_pddl, problem_pddl)
            
        print(f"Domain Name: {dom_name}")
        print(f"Initial State Atoms: {len(init_state.atoms)}")
        # print([str(a) for a in init_state.atoms])
        
        print(f"Found {len(succ)} successors")
        for act, state in succ:
            print(f"Action: {act}")
            
        if len(succ) == 0:
            print("\nVERIFICATION: No successors found. Checking why...")
            # Let's print the init state details to manual check
            print("Objects:")
            # We need to parse objects to see them
            parsed = unified_parser.parse_pddl_problem(problem_pddl)
            objects = parsed[0]
            print(json.dumps(objects, indent=2))
            
            print("Init State Predicates:")
            sorted_atoms = sorted([str(a) for a in init_state.atoms])
            for a in sorted_atoms:
                print(a)
                
    except Exception as e:
        print(f"Error during successor generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_specific_example()
