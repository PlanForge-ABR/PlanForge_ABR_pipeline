#!/usr/bin/env python3
"""
Evaluate plan generation results in new_results/planforge using KCL-Planning/VAL validator.
Matches predictions with ground-truth test datasets in data/test/*.json.
"""

import os
import json
import glob
import tempfile
import subprocess
from typing import Dict, Any, List

from evaluation.action_val import fix_pddl_domain_typing

VAL_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VAL", "build", "macos64", "Release", "bin", "Validate")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-detect planforge directory in new_results/
NEW_RESULTS_DIR = os.path.join(BASE_DIR, "new_results", "planforge")
if not os.path.exists(NEW_RESULTS_DIR):
    NEW_RESULTS_DIR = os.path.join(BASE_DIR, "new_results")

DATA_TEST_DIR = os.path.join(BASE_DIR, "data", "test")
SUMMARIES_DIR = os.path.join(BASE_DIR, "new_results", "evaluation_summaries")
os.makedirs(SUMMARIES_DIR, exist_ok=True)

def validate_plan_with_val(pddl_domain: str, pddl_problem: str, plan_actions: List[Any]) -> Dict[str, Any]:
    """Run VAL Validate binary on a domain, problem, and plan."""
    if not plan_actions:
        return {"valid": False, "reason": "Empty plan"}

    pddl_domain = fix_pddl_domain_typing(pddl_domain)
    plan_lines = []
    for act in plan_actions:
        if isinstance(act, str):
            tokens = act.strip().split()
            if tokens:
                action_name = tokens[0].lower()
                args = [t.lower() for t in tokens[1:]]
                plan_lines.append(f"({action_name} {' '.join(args)})" if args else f"({action_name})")
        elif isinstance(act, (list, tuple)):
            if act:
                action_name = str(act[0]).lower()
                args = [str(a).lower() for a in act[1:]]
                plan_lines.append(f"({action_name} {' '.join(args)})" if args else f"({action_name})")
                    
    if not plan_lines:
        return {"valid": False, "reason": "Could not parse plan actions"}

    plan_content = "\n".join(plan_lines)

    with tempfile.TemporaryDirectory() as tmpdir:
        dom_file = os.path.join(tmpdir, "domain.pddl")
        prob_file = os.path.join(tmpdir, "problem.pddl")
        plan_file = os.path.join(tmpdir, "plan.plan")

        with open(dom_file, "w", encoding="utf-8") as f: f.write(pddl_domain)
        with open(prob_file, "w", encoding="utf-8") as f: f.write(pddl_problem)
        with open(plan_file, "w", encoding="utf-8") as f: f.write(plan_content)

        cmd = [VAL_BIN, "-v", dom_file, prob_file, plan_file]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            stdout = res.stdout
            stderr = res.stderr
        except subprocess.TimeoutExpired:
            return {"valid": False, "reason": "VAL execution timeout"}
        except Exception as e:
            return {"valid": False, "reason": f"VAL execution error: {e}"}

        if "Plan valid" in stdout:
            return {"valid": True, "reason": "Plan valid"}
        elif "Bad plan description" in stdout or "Bad plan description" in stderr:
            return {"valid": False, "reason": "Bad plan description / syntax error"}
        elif "unsatisfied precondition" in stdout or "unsatisfied precondition" in stderr:
            return {"valid": False, "reason": "Unsatisfied precondition"}
        elif "Goal not satisfied" in stdout or "Goal not satisfied" in stderr:
            return {"valid": False, "reason": "Goal not satisfied"}
        elif "Plan failed to execute" in stdout or "Plan failed to execute" in stderr:
            return {"valid": False, "reason": "Plan failed to execute"}
        elif "Error in type-checking" in stdout or "Error in type-checking" in stderr:
            return {"valid": False, "reason": "Type-checking error"}
        else:
            first_line = stdout.strip().split("\n")[0] if stdout.strip() else stderr.strip().split("\n")[0] if stderr.strip() else "Unknown failure"
            return {"valid": False, "reason": f"Validation failed: {first_line[:80]}"}

def evaluate_all(results_dir: str = NEW_RESULTS_DIR, data_test_dir: str = DATA_TEST_DIR) -> Dict[str, Any]:
    result_files = sorted(glob.glob(os.path.join(results_dir, "*_test_results.json")))
    
    domain_summaries = {}
    total_all_examples = 0
    total_all_correct_decision = 0
    total_all_gt_yes = 0
    total_all_plan_verified = 0
    total_all_correct_with_plan = 0
    total_all_pred_yes = 0

    eval_details_by_domain = {}

    for rfile in result_files:
        fname = os.path.basename(rfile)
        domain = fname.replace("_test_results.json", "")
        
        with open(rfile, "r") as f:
            res_json = json.load(f)
            
        gt_file = os.path.join(data_test_dir, f"{domain}.json")
        if not os.path.exists(gt_file):
            continue
            
        with open(gt_file, "r") as f:
            gt_data = json.load(f)
        gt_map = {ex["id"]: ex for ex in gt_data}
        
        domain_instances = res_json.get("results", [])
        
        d_total = len(domain_instances)
        d_gt_yes = 0
        d_pred_yes = 0
        d_correct_decision = 0
        d_plan_verified = 0
        d_correct_with_plan = 0
        
        eval_examples = []
        
        for item in domain_instances:
            ex_id = item["id"]
            gt_ex = gt_map.get(ex_id, {})
            
            gt_ans = str(gt_ex.get("answer", "no")).strip().lower()
            pred_ans = str(item.get("plan_existence", "no")).strip().lower()
            plan = item.get("predicted_plan", [])
            
            pddl_domain = gt_ex.get("PDDL_domain", "")
            pddl_problem = gt_ex.get("PDDL_problem", "")
            
            is_gt_yes = (gt_ans == "yes")
            is_pred_yes = (pred_ans == "yes")
            is_correct_decision = (gt_ans == pred_ans)
            
            if is_gt_yes:
                d_gt_yes += 1
            if is_pred_yes:
                d_pred_yes += 1
            if is_correct_decision:
                d_correct_decision += 1
                
            val_res = {"valid": False, "reason": "N/A"}
            plan_valid = False
            
            if is_pred_yes and plan:
                val_res = validate_plan_with_val(pddl_domain, pddl_problem, plan)
                if val_res["valid"]:
                    plan_valid = True

            if is_gt_yes and plan_valid:
                d_plan_verified += 1
                
            is_correct_with_plan = False
            if is_gt_yes:
                if is_pred_yes and plan_valid:
                    is_correct_with_plan = True
            else:
                if not is_pred_yes:
                    is_correct_with_plan = True
                    
            if is_correct_with_plan:
                d_correct_with_plan += 1

            eval_examples.append({
                "id": ex_id,
                "gt_answer": gt_ans,
                "pred_answer": pred_ans,
                "correct_decision": is_correct_decision,
                "plan_valid": plan_valid,
                "val_reason": val_res["reason"],
                "correct_with_plan": is_correct_with_plan
            })

        plan_acc_yes = (d_plan_verified / d_gt_yes) if d_gt_yes > 0 else 0.0
        dec_acc = (d_correct_decision / d_total) if d_total > 0 else 0.0
        acc_with_plan = (d_correct_with_plan / d_total) if d_total > 0 else 0.0
        prec_plans = (d_plan_verified / d_pred_yes) if d_pred_yes > 0 else 0.0

        domain_summaries[domain] = {
            "total_examples": d_total,
            "gt_yes_examples": d_gt_yes,
            "pred_yes_examples": d_pred_yes,
            "plan_verified": d_plan_verified,
            "correct_decision": d_correct_decision,
            "correct_with_plan": d_correct_with_plan,
            "plan_accuracy_on_yes": plan_acc_yes,
            "decision_accuracy": dec_acc,
            "accuracy_with_plan": acc_with_plan,
            "precision_of_plans": prec_plans
        }

        eval_details_by_domain[domain] = eval_examples

        total_all_examples += d_total
        total_all_correct_decision += d_correct_decision
        total_all_gt_yes += d_gt_yes
        total_all_plan_verified += d_plan_verified
        total_all_correct_with_plan += d_correct_with_plan
        total_all_pred_yes += d_pred_yes

    overall_plan_acc_yes = (total_all_plan_verified / total_all_gt_yes) if total_all_gt_yes > 0 else 0.0
    overall_dec_acc = (total_all_correct_decision / total_all_examples) if total_all_examples > 0 else 0.0
    overall_acc_with_plan = (total_all_correct_with_plan / total_all_examples) if total_all_examples > 0 else 0.0
    overall_precision_plans = (total_all_plan_verified / total_all_pred_yes) if total_all_pred_yes > 0 else 0.0

    summary_data = {
        "overall": {
            "total_examples": total_all_examples,
            "total_gt_yes": total_all_gt_yes,
            "total_pred_yes": total_all_pred_yes,
            "total_plan_verified": total_all_plan_verified,
            "total_correct_decision": total_all_correct_decision,
            "total_correct_with_plan": total_all_correct_with_plan,
            "overall_plan_accuracy_on_yes": overall_plan_acc_yes,
            "overall_decision_accuracy": overall_dec_acc,
            "overall_accuracy_with_plan": overall_acc_with_plan,
            "overall_precision_of_plans": overall_precision_plans,
        },
        "per_domain": domain_summaries,
        "details": eval_details_by_domain
    }

    out_file = os.path.join(SUMMARIES_DIR, "val_evaluation_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nEvaluation summary saved to: {out_file}")
    return summary_data

if __name__ == "__main__":
    evaluate_all()
