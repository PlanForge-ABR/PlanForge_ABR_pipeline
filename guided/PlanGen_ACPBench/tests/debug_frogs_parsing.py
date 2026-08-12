
import json
import re
import sys
import time

def tokenize(pddl: str):
    # Remove comments
    pddl = re.sub(r";;?.*$", "", pddl, flags=re.MULTILINE)
    pddl = pddl.replace("\t", " ")
    # Find parentheses and non-whitespace sequences
    return re.findall(r"\(|\)|[^\s()]+", pddl)

try:
    with open('data/train/frogs_jumping-training.json', 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} examples")
    
    for i, ex in enumerate(data):
        print(f"Processing example {i}: {ex.get('id', 'unknown')}", end='\r')
        domain_pddl = ex.get('PDDL_domain')
        if not domain_pddl:
            continue
            
        start = time.time()
        tokens = tokenize(domain_pddl)
        dur = time.time() - start
        
        if dur > 1.0:
            print(f"\nExample {i} took {dur:.2f}s to tokenize!")
            
    print("\nDone processing all examples.")

except Exception as e:
    print(f"\nError: {e}")
