import os
import glob
import json
import re
import subprocess
import tempfile
import sys

project_root = "/Users/sarvesh/Desktop/IBM/IBM"
sys.path.append(project_root)
os.chdir(project_root)

from evaluation.action_val import fix_pddl_domain_typing

# Locate Validate binary
bin_paths = [
    "VAL/build/macos64/Release/bin/Validate",
    "/Users/sarvesh/Desktop/IBM/IBM/VAL/build/macos64/Release/bin/Validate"
]
validate_bin_path = None
for p in bin_paths:
    if os.path.exists(p):
        validate_bin_path = p
        break
if not validate_bin_path:
    validate_bin_path = "VAL/build/macos64/Release/bin/Validate"

# 1. Load PDDL domain and problem mapping from test_baseline
pddl_map = {}
for test_file in glob.glob("data/test_baseline/*.json"):
    with open(test_file, 'r') as f:
        data = json.load(f)
        for item in data:
            item_id = item.get("id")
            if item_id:
                pddl_map[item_id] = (item.get("PDDL_domain"), item.get("PDDL_problem"))

def format_step(step_str):
    step_str = step_str.strip().lower()
    if step_str.startswith("(") and step_str.endswith(")"):
        step_str = step_str[1:-1].strip()
    parts = step_str.split()
    if not parts:
        return ""
    action_name = parts[0]
    args = parts[1:]
    return f"({action_name} {' '.join(args)})"

def run_val(domain_pddl, problem_pddl, plan_steps):
    plan_content = "\n".join([format_step(s) for s in plan_steps if format_step(s)])
    with tempfile.TemporaryDirectory() as temp_dir:
        domain_file = os.path.join(temp_dir, "domain.pddl")
        problem_file = os.path.join(temp_dir, "problem.pddl")
        plan_file = os.path.join(temp_dir, "plan.plan")

        with open(domain_file, "w") as f:
            f.write(domain_pddl)
        with open(problem_file, "w") as f:
            f.write(problem_pddl)
        with open(plan_file, "w") as f:
            f.write(plan_content)

        cmd = [validate_bin_path, "-v", domain_file, problem_file, plan_file]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout = res.stdout
            if "Plan valid" in stdout:
                return True
        except Exception:
            pass
    return False

results = {}

domains = [
    "alfworld", "blocksworld", "depot", "ferry", "floortile", 
    "frogs_jumping", "goldminer", "grid", "grippers", "hanoi", 
    "logistics", "rovers", "satellite", "swap", "visitall"
]

for domain in domains:
    print(f"Evaluating domain: {domain} ...")
    if domain == "logistics":
        json_path = "PlanForge_autonomus results/logistics/full_15_results.json"
    else:
        json_path = f"PlanForge_autonomus results/{domain}.json"
        
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        
    if isinstance(raw_data, dict) and "rows" in raw_data:
        data = raw_data["rows"]
    elif isinstance(raw_data, dict) and "results" in raw_data:
        data = raw_data["results"]
    elif isinstance(raw_data, dict) and "details" in raw_data:
        data = raw_data["details"]
    elif isinstance(raw_data, list):
        data = raw_data
    else:
        print(f"Skipping unknown JSON root format for {domain}")
        continue
        
    total_examples = 0
    total_correct = 0
    total_correct_with_plan = 0
    total_plan_verified = 0
    total_plan_possible = 0
    
    for ex in data:
        if not isinstance(ex, dict):
            continue
            
        total_examples += 1
        
        ex_id = ex.get("instance_id") or ex.get("id")
        gt_answer = str(ex.get("gold_answer") or ex.get("expected_answer") or ex.get("answer") or "no").lower()
        pred_answer = str(ex.get("predicted_answer") or "no").lower()
        
        is_correct = (gt_answer == pred_answer)
        if is_correct:
            total_correct += 1
            
        plan_valid = False
        
        plan_steps = []
        if domain == "logistics":
            plan_file_field = ex.get("plan_file")
            if plan_file_field:
                basename = os.path.basename(plan_file_field.replace('\\', '/'))
                local_plan_path = os.path.join("PlanForge_autonomus results/logistics", basename)
                if os.path.exists(local_plan_path):
                    with open(local_plan_path, 'r') as pf:
                        for line in pf:
                            line = line.strip()
                            if line:
                                line_clean = re.sub(r"^\d+[\.\\)]\s*", "", line)
                                plan_steps.append(line_clean)
        else:
            plan_steps = ex.get("predicted_plan") or ex.get("plan") or ex.get("generated_plan") or []
            
        if pred_answer == "yes" and plan_steps:
            pddl = pddl_map.get(ex_id)
            if pddl:
                domain_pddl, problem_pddl = pddl
                domain_pddl = fix_pddl_domain_typing(domain_pddl)
                plan_valid = run_val(domain_pddl, problem_pddl, plan_steps)
                
        is_correct_with_plan = False
        if gt_answer == "yes":
            total_plan_possible += 1
            if plan_valid:
                total_plan_verified += 1
                if pred_answer == "yes":
                    is_correct_with_plan = True
        else:
            if pred_answer == "no":
                is_correct_with_plan = True
                
        if is_correct_with_plan:
            total_correct_with_plan += 1
            
    results[domain] = {
        "total_examples": total_examples,
        "total_correct": total_correct,
        "total_correct_with_plan": total_correct_with_plan,
        "total_plan_verified": total_plan_verified,
        "total_plan_possible": total_plan_possible
    }

print("\n\nPER-DOMAIN RESULTS:")
print(json.dumps(results, indent=2))

global_examples = 0
global_correct = 0
global_correct_with_plan = 0
global_plan_verified = 0
global_plan_possible = 0

for dom, stats in results.items():
    global_examples += stats["total_examples"]
    global_correct += stats["total_correct"]
    global_correct_with_plan += stats["total_correct_with_plan"]
    global_plan_verified += stats["total_plan_verified"]
    global_plan_possible += stats["total_plan_possible"]

overall_accuracy = global_correct / global_examples if global_examples else 0
overall_accuracy_with_plan = global_correct_with_plan / global_examples if global_examples else 0
overall_plan_accuracy = global_plan_verified / global_plan_possible if global_plan_possible else 0

summary = {
    "overall_accuracy": overall_accuracy,
    "overall_accuracy_with_plan": overall_accuracy_with_plan,
    "overall_plan_accuracy": overall_plan_accuracy,
    "total_examples": global_examples,
    "total_correct": global_correct,
    "total_correct_with_plan": global_correct_with_plan,
    "total_plan_verified": global_plan_verified,
    "total_plan_possible": global_plan_possible
}

print("\n\nOVERALL SUMMARY:")
print(json.dumps(summary, indent=2))

with open("autonomous_results.json", "w") as f:
    json.dump({"summary": summary, "domains": results}, f, indent=2)
