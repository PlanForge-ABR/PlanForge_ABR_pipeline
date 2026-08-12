
import sys
import os

# Add baselines to path
sys.path.append(os.path.join(os.getcwd(), "baselines"))
sys.path.append(os.getcwd())

from baselines.action_executor import get_executor

DOMAINS = [
    "blocksworld", "ferry", "logistics", "grippers", "rovers",
    "visitall", "grid", "floortile", "depot", "goldminer",
    "satellite", "swap", "alfworld"
]

# Valid PDDL action examples for each domain to test dispatch
# These arguments don't need to make logical sense vs state, just checking parsing/dispatch doesn't crash
DOMAIN_TESTS = {
    "blocksworld": ("pick-up(b1)", "pick-up", ["b1"]),
    "ferry": ("board(car1, loc1)", "board", ["car1", "loc1"]),
    "logistics": ("load-truck(obj1, trk1, loc1)", "load-truck", ["obj1", "trk1", "loc1"]),
    "grippers": ("pick(r1, ball1, room1, left)", "pick", ["r1", "ball1", "room1", "left"]),
    "rovers": ("navigate(rover1, wp1, wp2)", "navigate", ["rover1", "wp1", "wp2"]),
    "visitall": ("move(loc1, loc2)", "move", ["loc1", "loc2"]),
    "grid": ("move(pos1, pos2)", "move", ["pos1", "pos2"]),
    "floortile": ("up(robot1, tile1, tile2)", "up", ["robot1", "tile1", "tile2"]),
    "depot": ("drive(truck1, loc1, loc2)", "drive", ["truck1", "loc1", "loc2"]),
    "goldminer": ("move(loc1, loc2)", "move", ["loc1", "loc2"]),
    "satellite": ("turn_to(sat1, dir1, dir2)", "turn_to", ["sat1", "dir1", "dir2"]),
    "swap": ("swap(c1, c2, r1, r2)", "swap", ["c1", "c2", "r1", "r2"]),
    "alfworld": ("go_to_location(agent1, loc1, loc2)", "go_to_location", ["agent1", "loc1", "loc2"]),
}

def verify_all():
    print("Verifying strict execution for all domains...\n")
    failures = []
    
    for domain in DOMAINS:
        print(f"Testing {domain}...")
        try:
            executor = get_executor(domain)
            action_str, expected_name, expected_args = DOMAIN_TESTS[domain]
            
            # 1. Test try_parse_action
            parsed = executor.try_parse_action(action_str)
            if not parsed:
                print(f"  [FAIL] try_parse_action failed for '{action_str}'")
                failures.append(f"{domain}: parse failed")
                continue
                
            name, args = parsed
            if name != expected_name or args != expected_args:
                print(f"  [FAIL] Parsed incorrectly. Got ({name}, {args}), expected ({expected_name}, {expected_args})")
                failures.append(f"{domain}: parse mismatch")
                continue
                
            print(f"  [PASS] Parsed '{action_str}' -> {name}{args}")
            
            # 2. Test _execute_action dispatch (mocking helpers)
            # We want to ensure _execute_action calls the right helper and passes (args, facts)
            # We'll rely on the fact that if we pass invalid arguments (state-wise), it returns False
            # but usually shouldn't crash if signatures are correct.
            # However, some helpers check len(args) and might return False safely.
            # We just want to ensure NO CRASH.
            
            try:
                # We pass empty facts. Helpers might fail logic but shouldn't crash on TypeError
                executor._execute_action(name, args, []) 
                print(f"  [PASS] dispatched {name} without crash")
            except TypeError as e:
                print(f"  [FAIL] TypeError during dispatch of {name}: {e}")
                failures.append(f"{domain}: dispatch TypeError {e}")
            except Exception as e:
                print(f"  [FAIL] Exception during dispatch of {name}: {e}")
                failures.append(f"{domain}: dispatch Exception {e}")
                
        except Exception as e:
            print(f"  [FAIL] Executor init failed: {e}")
            failures.append(f"{domain}: init failed {e}")
            
    print("\n--- Summary ---")
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)
    else:
        print("ALL DOMAINS PASSED strict execution checks.")

if __name__ == "__main__":
    verify_all()
