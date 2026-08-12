
import sys
import os
import json
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search import Search

def test_search_timeout():
    print("Testing Search Timeout...")
    # Blocksworld simple problem
    initial_state = [{"predicate": "on-table", "args": ["a"]}, {"predicate": "on-table", "args": ["b"]}, {"predicate": "clear", "args": ["a"]}, {"predicate": "clear", "args": ["b"]}, {"predicate": "arm-empty", "args": []}]
    goal_state = [{"predicate": "on", "args": ["a", "b"]}]
    
    initial_json = json.dumps(initial_state)
    goal_json = json.dumps(goal_state)
    
    # Set timeout to 0.01 seconds which should trigger timeout for any reasonable search
    # We might need a slightly complex problem or artificially slow down search to ensure timeout 
    # but for now let's try extremely short timeout 
    searcher = Search("blocksworld", initial_json, goal_json, timeout=0.0001)
    
    found, plan, info = searcher.search(strategy="astar", return_plan=True)
    
    print(f"Result: found={found}, info={info}")
    
    if not found and info.get("reason") == "timeout":
        print("✅ Timeout test passed")
    elif found:
         print("⚠️ Search finished too quickly to timeout, try reducing timeout or increasing complexity")
    else:
        print(f"❌ Timeout test failed. Reason: {info.get('reason')}")

def test_search_exhausted():
    print("\nTesting Search Exhausted...")
    # Impossible goal in blocksworld (e.g. object floating without support, though domain rules might just not generate it)
    # Better: a graph that is small but goal is unreachable.
    # Let's say we have only block 'a', goal is on(a, b). 'b' implies existence but if 'b' is not in initial state logic might break or just fail.
    # Actually, simpler: initial: on-table(a), clear(a), arm-empty. Goal: on(a, a) (impossible standardly)
    
    initial_state = [{"predicate": "on-table", "args": ["a"]}, {"predicate": "clear", "args": ["a"]}, {"predicate": "arm-empty", "args": []}]
    goal_state = [{"predicate": "on", "args": ["a", "a"]}]
    
    initial_json = json.dumps(initial_state)
    goal_json = json.dumps(goal_state)
    
    searcher = Search("blocksworld", initial_json, goal_json, timeout=5)
    
    found, plan, info = searcher.search(strategy="bfs", return_plan=True)
    
    print(f"Result: found={found}, info={info}")
    
    if not found and info.get("reason") == "exhausted":
        print("✅ Exhausted test passed")
    else:
        print(f"❌ Exhausted test failed. Reason: {info.get('reason')}")

if __name__ == "__main__":
    try:
        test_search_timeout()
        test_search_exhausted()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
