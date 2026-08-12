#!/usr/bin/env python3
"""
Re-run search for domains using existing NL2State results from search_result_test_domain.json.
This is useful when heuristics have been updated but we don't want to re-run the expensive NL2State step.
"""

import argparse
import json
import os
import sys
import time
from src.search import Search
from src import results
import concurrent.futures



def solve_single_example(args):
    domain, item, timeout, strategy = args
    example_id = item.get('example_id')
    # print(f"Starting search for {example_id}...") # Optional: noise reduction
    
    initial_state_str = item.get('predicted_initial_state')
    goal_state_str = item.get('predicted_goal_state')
    
    if not initial_state_str or not goal_state_str:
        return item
        
    try:
        searcher = Search(domain, initial_state_str, goal_state_str, timeout=timeout)
        found, plan = searcher.search(strategy=strategy, return_plan=True)
        
        item['plan'] = plan
        item['final_answer'] = 'yes' if found else 'no'
        print(f"[{example_id}] Result: {item['final_answer']}", flush=True)
        
    except Exception as e:
        print(f"[{example_id}] Error: {e}", flush=True)
        pass
        
    return item

def rerun_search(domain, src_root='src', timeout=60, strategy='greedy_bfs', example_id_filter=None):
    result_path = os.path.join(src_root, domain, "search_result_test_domain.json")
    if not os.path.exists(result_path):
        print(f"Error: {result_path} not found.")
        return

    print(f"Loading previous results from {result_path}...")
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    # Filter by specific example_id if provided
    if example_id_filter:
        original_count = len(data)
        data = [d for d in data if d.get('example_id') == example_id_filter]
        print(f"Filtering for example_id={example_id_filter}, found {len(data)} matches out of {original_count}")

    print(f"Re-running search for {len(data)} examples in {domain} with {os.cpu_count()} workers...")
    
    tasks = [(domain, item, timeout, strategy) for item in data]
    
    updated_data = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use map to preserve order if needed, or just iterate futures
        # We'll use map here
        futures = {executor.submit(solve_single_example, task): task for task in tasks}
        search_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                # We can get the result immediately
                res = future.result()
                search_results.append(res)
            except Exception as e:
                task = futures[future]
                print(f"Task generated an exception: {e}")
        
        # Sort results back to original order to avoid randomization
        # We can map back using example_id
        result_map = {item['example_id']: item for item in search_results}
        updated_data = [result_map.get(task[1]['example_id'], task[1]) for task in tasks]

    print(f"Saving updated results to {result_path}...")
    with open(result_path, 'w') as f:
        json.dump(updated_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domains', nargs='+', help='List of domains to re-run')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--strategy', default='greedy_bfs')
    parser.add_argument('--src', default='src')
    parser.add_argument('--example_id', help='Specific example ID to run')
    args = parser.parse_args()

    for domain in args.domains:
        rerun_search(domain, src_root=args.src, timeout=args.timeout, strategy=args.strategy, example_id_filter=args.example_id)
        
    print("\nRe-computing summaries...")
    # We need to know where the ground truth is data/test_baseline
    # We can invoke results.py with the right arguments
    # Assuming test split
    
    # Python call to results.main might be cleaner but subprocess is easier to match arguments
    # We use the fixed results.py which handles data_root
    for domain in args.domains:
        # results.py runs per domain automatically if we just re-run the summary generation for 'test'
        # BUT results.py iterates over ALL folders in src matching.
        pass

    import subprocess
    cmd = [
        sys.executable, "src/results.py",
        "--split", "test",
        "--data_root", "data/test_baseline",
        "--out", "search_result_test_domain.json",
        "--summary_out", "accuracy_summary_test_domain.json",
        "--dom_summary_out", "search_accuracy_test_domain.json"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
