#!/usr/bin/env python3
"""
Test harness for any domain successor and is_goal functions.
"""
import json
import sys
import importlib.util
from pathlib import Path

DOMAIN = 'depots'

def load_function(file_path, func_name):
    """Dynamically load a function from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location(func_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)
    except Exception as e:
        print(f"Error loading {func_name} from {file_path}: {e}")
        return None

def normalize_state(state):
    """Normalize a state for comparison by sorting predicates."""
    return sorted(
        [{"predicate": p["predicate"], "args": p["args"]} for p in state],
        key=lambda x: (x["predicate"], tuple(x["args"]))
    )

def states_equal(state1, state2):
    """Check if two states are equal (order-independent)."""
    return normalize_state(state1) == normalize_state(state2)

def test_successor():
    """Test the successor function."""
    print("=" * 60)
    print("TESTING SUCCESSOR FUNCTION")
    print("=" * 60)
    
    succ_func = load_function('succ.py', 'successor')
    if not succ_func:
        print("Failed to load successor function!")
        return False
    
    with open('succ_tests.json') as f:
        data = json.load(f)
    tests = data[DOMAIN]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests):
        try:
            result = succ_func(test['initial_state'])
            
            # Prepare expected data: set of (action, frozenset_of_normalized_state_str)
            expected_set = set()
            for ns in test['next_states']:
                norm_state = normalize_state(ns['state'])
                # Convert normalized state to a stable string representation for set comparison
                state_str = json.dumps(norm_state, sort_keys=True)
                expected_set.add((ns['action'], state_str))
            
            # Process result data
            result_set = set()
            for item in result:
                # Check format
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError(f"Expected (action, state) tuple, got {type(item)}")
                
                action, state = item
                if not isinstance(action, str):
                     raise ValueError(f"Expected action to be string, got {type(action)}")

                norm_state = normalize_state(state)
                state_str = json.dumps(norm_state, sort_keys=True)
                result_set.add((action, state_str))
            
            # Check if sets are equal
            if result_set == expected_set:
                passed += 1
                print(f"✓ Test {i+1}/{len(tests)} passed: {test['example_id']}")
            else:
                failed += 1
                print(f"\n✗ Test {i+1}/{len(tests)} FAILED: {test['example_id']}")
                print(f"  Expected {len(expected_set)} (action, state) pairs")
                print(f"  Got {len(result_set)} (action, state) pairs")
                
                # Verify if we at least got the right states (ignoring actions for debug info)
                result_states = {s for _, s in result_set}
                expected_states = {s for _, s in expected_set}
                if result_states == expected_states:
                    print("  Note: States match, but actions differ or are missing.")
                else:
                    print(f"  States mismatch as well. Expected {len(expected_states)}, got {len(result_states)}")

        except Exception as e:
            failed += 1
            print(f"\n✗ Test {i+1}/{len(tests)} EXCEPTION: {test['example_id']}")
            print(f"  Error: {e}")
            # print traceback for debugging if needed, but keeping it clean for now
            # import traceback
            # traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Successor Tests: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}\n")
    return failed == 0

def test_is_goal():
    """Test the is_goal function."""
    print("=" * 60)
    print("TESTING IS_GOAL FUNCTION")
    print("=" * 60)
    
    is_goal_func = load_function('is_goal.py', 'is_goal')
    if not is_goal_func:
        print("Failed to load is_goal function!")
        return False
    
    with open('goal_tests.json') as f:
        data = json.load(f)
    tests = data[DOMAIN]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests):
        try:
            result = is_goal_func(test['current_state'], test['goal_state'])
            expected = test['expected']
            
            if result == expected:
                passed += 1
                print(f"✓ Test {i+1}/{len(tests)} passed: {test['example_id']}")
            else:
                failed += 1
                print(f"\n✗ Test {i+1}/{len(tests)} FAILED: {test['example_id']}")
                print(f"  Expected: {expected}, Got: {result}")
                
        except Exception as e:
            failed += 1
            print(f"\n✗ Test {i+1}/{len(tests)} EXCEPTION: {test['example_id']}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Goal Tests: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}\n")
    return failed == 0

if __name__ == "__main__":
    all_passed = True
    all_passed = test_is_goal() and all_passed
    all_passed = test_successor() and all_passed
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED ❌\n")
        sys.exit(1)

