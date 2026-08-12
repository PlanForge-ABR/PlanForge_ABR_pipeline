
import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Any

def get_plan_length(example: Dict[str, Any]) -> int:
    """Extract or calculate plan length from example."""
    # 1. Trust 'plan_length' if it is an integer
    pl = example.get('plan_length')
    if isinstance(pl, int):
        return pl
        
    # 2. explicit 'no' answer often implies 0 length (unsolvable)
    if example.get('answer') == 'no':
        return 0

    # 3. Try to parse 'sample_plan' or 'optimal_plan'
    plan = example.get('optimal_plan') or example.get('sample_plan') or example.get('plan') or example.get('solution')
    
    if isinstance(plan, list):
        return len(plan)
    elif isinstance(plan, str) and plan.strip():
        # formatted like "(action 1) (action 2)" or "action1, action2"
        if '(' in plan:
            return plan.count('(')
        return len(plan.strip().split('\n'))
    
    # If no plan provided and answer is yes (or unknown), treat as unknown -1
    return -1

def create_stratified_test_set(domain: str, train_root: str, test_root: str, n_per_length: int = 5):
    """Create a stratified test set for the domain."""
    train_file = os.path.join(train_root, f"{domain}-training.json")
    
    if not os.path.exists(train_file):
        print(f"Warning: Training file not found for {domain}: {train_file}")
        # Try alternate name
        train_file = os.path.join(train_root, f"{domain.replace('_', '-')}-training.json")
        if not os.path.exists(train_file):
            print(f"Error: Could not find training data for {domain}")
            return

    with open(train_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} training examples for {domain}")
    
    # Group by length
    by_length = defaultdict(list)
    for ex in data:
        length = get_plan_length(ex)
        by_length[length].append(ex)
        
    # Sample from smallest 3 lengths only
    valid_lengths = sorted(by_length.keys())[:3]
    print(f"Stratification for {domain} (limiting to lengths {valid_lengths}):")
    
    test_set = []
    for length in valid_lengths:
        examples = by_length[length]
        # Shuffle to get random samples if we have enough
        random.shuffle(examples)
        
        # Take n_per_length, or all if fewer
        selected = examples[:n_per_length]
        test_set.extend(selected)
        print(f"  Length {length}: {len(examples)} available -> {len(selected)} selected")
        
    # Verify we have something
    if not test_set:
        print(f"Warning: No test examples generated for {domain}")
        return
        
    # Save
    os.makedirs(test_root, exist_ok=True)
    out_file = os.path.join(test_root, f"{domain}-test.json")
    
    # The evaluation script expects a list of dicts.
    # We should ensure the format matches what evaluate.py expects.
    # evaluate.py reads 'context', 'question'/'inputs', 'answer'.
    # Training data usually has these.
    
    # We should also normalize the ID to avoid key collision if we merge later?
    # No, keep original IDs.
    
    with open(out_file, 'w', encoding="utf-8") as f:
        json.dump(test_set, f, indent=2)
        
    print(f"Saved {len(test_set)} test examples to {out_file}")

if __name__ == "__main__":
    TRAIN_ROOT = "data/train"
    TEST_ROOT = "data/test_baseline"
    DOMAINS = ["frogs_jumping", "hanoi"]
    
    for dom in DOMAINS:
        create_stratified_test_set(dom, TRAIN_ROOT, TEST_ROOT)
