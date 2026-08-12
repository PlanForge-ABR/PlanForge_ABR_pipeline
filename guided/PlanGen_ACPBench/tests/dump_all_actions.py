
import sys
import os

# Add baselines to path
sys.path.append(os.path.join(os.getcwd(), "baselines"))
sys.path.append(os.getcwd())

from baselines.zero_shot_dspy import load_pddl_domain, PDDLParser

DOMAINS = [
    "blocksworld", "ferry", "logistics", "grippers", "rovers",
    "visitall", "grid", "floortile", "depot", "goldminer",
    "satellite", "swap", "alfworld"
]

def dump_actions():
    print("Dumping PDDL actions for all domains...\n")
    for domain in DOMAINS:
        print(f"--- {domain.upper()} ---")
        pddl_str = load_pddl_domain(domain)
        if not pddl_str:
            print(f"  [ERROR] Could not load PDDL for {domain}")
            continue
        
        parser = PDDLParser(pddl_str)
        if not parser.actions:
            print("  [WARN] No actions found parsed.")
            continue
            
        for name, action in sorted(parser.actions.items()):
            params = ", ".join(action.parameters)
            print(f"  {name}({params})")
        print("")

if __name__ == "__main__":
    dump_actions()
