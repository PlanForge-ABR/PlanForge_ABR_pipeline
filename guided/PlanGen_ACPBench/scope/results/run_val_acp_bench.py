#!/usr/bin/env python3
import os
import re
import json
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional, Any

# Resolve absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
validate_bin_path = os.path.join(project_root, "VAL/build/macos64/Release/bin/Validate")

def parse_action_string(action_str: str) -> Optional[Tuple[str, List[str]]]:
    """Helper to parse action(args) or (action args) or action args."""
    if not action_str:
        return None
    action_str = action_str.strip()
    # name(arg1, arg2)
    match = re.match(r"(\w[\w\-]*)\((.*)\)", action_str)
    if match:
        name = match.group(1)
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        return name, args
    # (name arg1 arg2)
    if action_str.startswith("(") and action_str.endswith(")"):
        parts = action_str[1:-1].split()
        if parts:
            return parts[0], parts[1:]
    # Simple name arg1 arg2
    parts = action_str.split()
    if parts:
        return parts[0], parts[1:]
    return None

def parse_actions_syntax(action_str: str) -> List[Tuple[str, List[str]]]:
    """Parse action sequences from a raw string (if plan is a string)."""
    if not action_str or not isinstance(action_str, str):
        return []

    def parse_expr(expr: str) -> List[Tuple[str, List[str]]]:
        expr = expr.strip()
        if not expr:
            return []
        while expr.startswith("((") and expr.endswith("))"):
            expr = expr[1:-1].strip()
        if expr.startswith("(") and expr.endswith(")"):
            inner = expr[1:-1].strip()
        else:
            inner = expr
        if not inner:
            return []
        if inner.lower().startswith("and "):
            body = inner[3:].strip()
            results: List[Tuple[str, List[str]]] = []
            balance = 0
            current: List[str] = []
            for ch in body:
                if ch == "(":
                    balance += 1
                elif ch == ")":
                    balance -= 1
                current.append(ch)
                if balance == 0 and ch == ")" and "".join(current).strip():
                    results.extend(parse_expr("".join(current).strip()))
                    current = []
            return results
        if inner.lower().startswith("not "):
            return []
        parts = inner.split()
        if not parts:
            return []
        return [(parts[0], parts[1:])]

    actions: List[Tuple[str, List[str]]] = []
    found_parens = False
    i = 0
    n = len(action_str)
    while i < n:
        if action_str[i] == "(":
            found_parens = True
            start = i
            depth = 1
            i += 1
            while i < n and depth > 0:
                if action_str[i] == "(":
                    depth += 1
                elif action_str[i] == ")":
                    depth -= 1
                i += 1
            actions.extend(parse_expr(action_str[start:i]))
        else:
            i += 1

    if found_parens and actions:
        return actions

    # Fallback: line-based parsing
    for line in action_str.split("\n"):
        line = line.strip()
        line = re.sub(r"^\d+[\.\\)]\s*", "", line)
        line = re.sub(r"^-\s*", "", line)
        if not line:
            continue
        line = line.replace('"', "").replace("'", "").replace("[", "").replace("]", "").replace(",", "")
        parts = line.split()
        if parts:
            actions.append((parts[0], parts[1:]))
    return actions

def verify_plan_with_val(
    pddl_domain: str,
    pddl_problem: str,
    plan: Any,
    debug: bool = False
) -> Dict[str, Any]:
    # 1. Parse actions
    parsed_actions = []
    if not isinstance(plan, list):
        if isinstance(plan, str):
            parsed_actions = parse_actions_syntax(plan)
    else:
        for action_item in plan:
            if isinstance(action_item, str):
                parsed = parse_action_string(action_item)
                if parsed:
                    parsed_actions.append(parsed)
            elif isinstance(action_item, dict):
                name = action_item.get("name") or action_item.get("action")
                args = action_item.get("args") or action_item.get("parameters")
                if name:
                    parsed_actions.append((name, args or []))
            elif isinstance(action_item, (list, tuple)):
                if len(action_item) >= 1:
                    parsed_actions.append((action_item[0], action_item[1] if len(action_item) > 1 else []))

    if debug:
        print(f"Parsed actions ({len(parsed_actions)}): {parsed_actions}")

    if not parsed_actions:
        return {
            "valid": False,
            "reason": "No actions could be parsed from plan",
            "actions_applied": 0,
            "total_actions": 0
        }

    # 2. Format plan steps
    plan_lines = []
    for name, args in parsed_actions:
        name_clean = name.lower()
        args_clean = [a.lower() for a in args]
        if args_clean:
            plan_lines.append(f"({name_clean} {' '.join(args_clean)})")
        else:
            plan_lines.append(f"({name_clean})")
    plan_content = "\n".join(plan_lines)

    # 3. Write to temp files and run Validate
    with tempfile.TemporaryDirectory() as temp_dir:
        domain_file = os.path.join(temp_dir, "domain.pddl")
        problem_file = os.path.join(temp_dir, "problem.pddl")
        plan_file = os.path.join(temp_dir, "plan.plan")

        with open(domain_file, "w") as f:
            f.write(pddl_domain)
        with open(problem_file, "w") as f:
            f.write(pddl_problem)
        with open(plan_file, "w") as f:
            f.write(plan_content)

        cmd = [validate_bin_path, "-v", domain_file, problem_file, plan_file]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout = res.stdout
            stderr = res.stderr
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Execution of Validate failed: {e}",
                "actions_applied": 0,
                "total_actions": len(parsed_actions)
            }

        # 4. Parse VAL output
        if "Plan valid" in stdout:
            return {
                "valid": True,
                "reason": "Goal satisfied",
                "actions_applied": len(parsed_actions),
                "total_actions": len(parsed_actions)
            }
        
        if "Error in type-checking" in stdout or "Error in type-checking" in stderr:
            return {
                "valid": False,
                "reason": "Error: Error in type-checking!",
                "actions_applied": 0,
                "total_actions": len(parsed_actions)
            }

        if "unsatisfied precondition at time" in stdout or "unsatisfied precondition at time" in stderr:
            match = re.search(r"unsatisfied precondition at time (\d+)", stdout + stderr)
            step = 1
            if match:
                step = int(match.group(1))
            
            failed_action_match = re.search(r"Plan failed because of unsatisfied precondition in:\n([^\n]+)", stdout)
            failed_action_info = ""
            if failed_action_match:
                failed_action_info = failed_action_match.group(1).strip()
            
            reason = f"Action {step} failed: unsatisfied precondition"
            if failed_action_info:
                reason = f"Action {step} '{failed_action_info}' failed: unsatisfied precondition"
                
            return {
                "valid": False,
                "reason": reason,
                "actions_applied": max(0, step - 1),
                "total_actions": len(parsed_actions)
            }

        if "Goal not satisfied" in stdout:
            return {
                "valid": False,
                "reason": "Goal not satisfied",
                "actions_applied": len(parsed_actions),
                "total_actions": len(parsed_actions)
            }

        if "Plan failed to execute" in stdout:
            return {
                "valid": False,
                "reason": "Plan failed to execute",
                "actions_applied": 0,
                "total_actions": len(parsed_actions)
            }

        return {
            "valid": False,
            "reason": f"Validation failed. Output: {stdout.strip()[:100]}",
            "actions_applied": 0,
            "total_actions": len(parsed_actions)
        }

def run_validation():
    # Directories
    results_dir = os.path.join(project_root, "scope/results/acp_bench")
    output_dir = os.path.join(results_dir, "output_full_gpt-5.4")
    baselines_dir = os.path.join(project_root, "data/test_baseline")
    summary_path = os.path.join(results_dir, "accuracy_summary.json")

    # Load summary to edit later
    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    overall_total = 0
    overall_plan_val_correct = 0

    # Locate all scope files
    scope_files = sorted([f for f in os.listdir(output_dir) if f.endswith("_scope.json")])
    print(f"Found {len(scope_files)} scope output files to validate.")

    for sf in scope_files:
        domain = sf.replace("_scope.json", "")
        sf_path = os.path.join(output_dir, sf)
        
        # Load scope output
        with open(sf_path, "r") as f:
            scope_data = json.load(f)

        # Load corresponding baseline file
        baseline_file = os.path.join(baselines_dir, f"{domain}-test.json")
        if not os.path.exists(baseline_file):
            print(f"Warning: baseline file {baseline_file} not found for domain {domain}")
            continue

        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)

        # Build mapping of id -> PDDL
        baseline_map = {ex["id"]: ex for ex in baseline_data if "id" in ex}

        domain_total = 0
        domain_plan_val_correct = 0

        print(f"\nValidating domain: {domain} ({len(scope_data)} examples)...")

        for ex in scope_data:
            ex_id = ex.get("example_id") or ex.get("id")
            if not ex_id:
                continue

            domain_total += 1
            overall_total += 1

            gt_answer = str(ex.get("ground_truth_answer") or ex.get("gt_answer") or "no").lower()
            pred_answer = str(ex.get("final_answer") or ex.get("pred_answer") or "no").lower()
            plan = ex.get("plan")

            plan_valid = False
            verification = {"valid": None, "reason": "N/A"}

            # We validate the plan if pred_answer is "yes" and plan is provided
            if pred_answer == "yes" and plan:
                baseline_ex = baseline_map.get(ex_id)
                if not baseline_ex:
                    print(f"  Warning: Example ID {ex_id} not found in baseline map.")
                    verification = {"valid": False, "reason": "Example ID not found in baseline data"}
                else:
                    pddl_domain = baseline_ex.get("PDDL_domain")
                    pddl_problem = baseline_ex.get("PDDL_problem")
                    if not pddl_domain or not pddl_problem:
                        verification = {"valid": False, "reason": "PDDL domain or problem missing from baseline"}
                    else:
                        verify_res = verify_plan_with_val(pddl_domain, pddl_problem, plan)
                        verification = verify_res
                        if verify_res["valid"]:
                            plan_valid = True

            # Save validation info to the example
            if pred_answer != "yes":
                ex["plan_valid"] = "N/A"
            else:
                ex["plan_valid"] = plan_valid
            ex["verification"] = verification
            ex["plan_failure_reason"] = verification.get("reason", "N/A")

            # Determine correctness with plan validation
            is_correct_with_plan = False
            if gt_answer == "yes":
                if pred_answer == "yes" and plan_valid:
                    is_correct_with_plan = True
            else: # gt_answer == "no"
                if pred_answer == "no":
                    is_correct_with_plan = True

            ex["correct_with_plan"] = is_correct_with_plan
            if is_correct_with_plan:
                domain_plan_val_correct += 1
                overall_plan_val_correct += 1

        # Write the updated scope file back
        with open(sf_path, "w") as f:
            json.dump(scope_data, f, indent=2)

        # Update domain-level stats in summary data
        domain_acc = f"{(domain_plan_val_correct / domain_total * 100):.2f}%" if domain_total else "0.00%"
        print(f"  Domain Accuracy (original): {summary_data['domains'][domain].get('accuracy')}")
        print(f"  Domain VAL Correct: {domain_plan_val_correct}/{domain_total} ({domain_acc})")

        summary_data['domains'][domain]['plan_val_correct'] = domain_plan_val_correct
        summary_data['domains'][domain]['plan_val_accuracy'] = domain_acc

    # Update metadata overall stats in summary data
    overall_acc = f"{(overall_plan_val_correct / overall_total * 100):.2f}%" if overall_total else "0.00%"
    print(f"\nOverall Stats:")
    print(f"  Overall Total: {overall_total}")
    print(f"  Overall VAL Correct: {overall_plan_val_correct} ({overall_acc})")

    summary_data['metadata']['overall_plan_val_correct'] = overall_plan_val_correct
    summary_data['metadata']['overall_plan_val_accuracy'] = overall_acc

    # Write the summary file back
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary updated and saved to {summary_path}")

if __name__ == "__main__":
    run_validation()
