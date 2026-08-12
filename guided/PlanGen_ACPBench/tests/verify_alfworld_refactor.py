
import sys
import os
import re

# Add baselines to path
sys.path.append(os.path.join(os.getcwd(), "baselines"))
sys.path.append(os.getcwd())

from baselines.action_executor import get_executor
from baselines.zero_shot_dspy import get_available_actions, load_pddl_domain, PDDLParser

def test_dynamic_pddl_loading():
    print("Testing dynamic PDDL loading for alfworld...")
    domain = "alfworld"
    pddl_str = load_pddl_domain(domain)
    if not pddl_str:
        print("[FAIL] Could not load PDDL for alfworld")
        return
    
    actions_desc = get_available_actions(domain)
    print(f"Loaded actions description (first 100 chars): {actions_desc[:100]}...")
    
    if "go_to_location" in actions_desc:
        print("[PASS] 'go_to_location' found in generated action description.")
    else:
        print("[FAIL] 'go_to_location' NOT found in generated action description.")

    if "pickup_object_from" in actions_desc:
        print("[PASS] 'pickup_object_from...' found.")
    else:
        print("[FAIL] 'pickup_object_from...' NOT found.")

def test_strict_action_execution():
    print("\nTesting strict action execution for AlfworldExecutor...")
    executor = get_executor("alfworld")
    
    # Test valid action parsing
    valid_action = "go_to_location(agent1, loc1, loc2)"
    parsed = executor.try_parse_action(valid_action)
    print(f"Parsing '{valid_action}': {parsed}")
    if parsed and parsed[0] == "go_to_location":
        print("[PASS] Parsed strict action name correctly.")
    else:
        print(f"[FAIL] Failed to parse strict action. Got: {parsed}")

    # Test invalid/fuzzy action parsing (should fail or return None/different?)
    # NOTE: try_parse_action regex is simple, so "go_to(agent1)" might parse as ("go_to", args).
    # But dispatch should fail.
    
    # Mocking _go_to to verify dispatch
    original_go_to = executor._go_to
    executor._go_to = lambda args, facts: True # Mock return True
    
    # Dispatch valid
    success = executor._execute_action("go_to_location", ["agent1", "l1", "l2"], [])
    if success:
        print("[PASS] Dispatch 'go_to_location' succeeded.")
    else:
        print("[FAIL] Dispatch 'go_to_location' failed.")
        
    # Dispatch invalid name (fuzzy)
    success_fuzzy = executor._execute_action("go_to", ["agent1", "l1", "l2"], [])
    if not success_fuzzy:
        print("[PASS] Dispatch 'go_to' failed as expected (strict matching).")
    else:
        print("[FAIL] Dispatch 'go_to' succeeded unexpectedly.")

    # Restore
    executor._go_to = original_go_to

def test_helper_signatures():
    print("\nTesting helper signatures (checking for TypeError)...")
    executor = get_executor("alfworld")
    
    # We just want to ensure calling them with (args, facts) doesn't raise TypeError
    # We'll pas empty args/facts which might return False/True but shouldn't crash
    
    try:
        executor._open([], [])
        print("[PASS] _open accepts (args, facts)")
    except TypeError as e:
        print(f"[FAIL] _open raised TypeError: {e}")

    try:
        executor._pickup([], [])
        print("[PASS] _pickup accepts (args, facts)")
    except TypeError as e:
        print(f"[FAIL] _pickup raised TypeError: {e}")

    try:
        executor._validate([], "clean")
        print("[PASS] _validate accepts (args, type_str)")
    except TypeError as e:
        print(f"[FAIL] _validate raised TypeError: {e}")
        
if __name__ == "__main__":
    try:
        test_dynamic_pddl_loading()
        test_strict_action_execution()
        test_helper_signatures()
    except Exception as e:
        print(f"\n[FATAL] Script crashed: {e}")
        import traceback
        traceback.print_exc()
