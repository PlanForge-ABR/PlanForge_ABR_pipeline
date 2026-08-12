#!/usr/bin/env python3
"""
SCOPE Baseline Evaluator for ACPBench.
Reads test data from data/ACPBench_dataset_final/test and saves results to new_results/.
Features real-time progress logging, token usage tracking, and exact USD cost recording.
"""

import os
import sys
import glob
import json
import time
import argparse
import logging
import tempfile
import subprocess
from typing import Dict, Any, List, Optional

# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from scope.workflow_acp import SCOPESolverBuilder, StandaloneSCOPESolver, token_usage
except ImportError:
    from workflow_acp import SCOPESolverBuilder, StandaloneSCOPESolver, token_usage

from evaluation.action_val import fix_pddl_domain_typing

VAL_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VAL", "build", "macos64", "Release", "bin", "Validate")

def _normalize_yes(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"yes", "y", "true", "t", "1"}

def find_exemplar(test_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Search for the first positive case ('yes') with a non-empty sample plan."""
    for example in test_data:
        ans = example.get("answer")
        plan = example.get("sample_plan")
        if _normalize_yes(ans) and isinstance(plan, list) and len(plan) > 0:
            return example
    for example in test_data:
        plan = example.get("sample_plan")
        if isinstance(plan, list) and len(plan) > 0:
            return example
    return None

def validate_plan_with_val(pddl_domain: str, pddl_problem: str, plan_actions: List[Any]) -> bool:
    """Run VAL Validate binary on a domain, problem, and plan."""
    if not plan_actions:
        return False
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
        return False
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
            return "Plan valid" in res.stdout
        except Exception:
            return False

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost based on GPT-5.4 token pricing ($2.50/1M input, $15.00/1M output)."""
    return (prompt_tokens * 2.50 + completion_tokens * 15.00) / 1000000.0

def main():
    parser = argparse.ArgumentParser(description="Run SCOPE on ACPBench dataset.")
    parser.add_argument("--domain", default="all", help="Domain to evaluate (e.g. blocksworld, ferry) or 'all'.")
    parser.add_argument("--test_dir", default="data/ACPBench_dataset_final/test", help="Path to ACPBench test dataset folder.")
    parser.add_argument("--output_dir", default="new_results/scope", help="Directory to save output results.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test examples per domain.")
    parser.add_argument("--no_validate", dest="validate", action="store_false", help="Disable automatic VAL validation.")
    parser.set_defaults(validate=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.domain.lower() == "all":
        domain_files = sorted(glob.glob(os.path.join(args.test_dir, "*.json")))
        domains = [os.path.basename(f).replace(".json", "") for f in domain_files]
    else:
        domains = [d.strip() for d in args.domain.split(",")]

    print(f"\n" + "=" * 85)
    print(f" 🚀 STARTING SCOPE EVALUATION ON {len(domains)} DOMAIN(S)")
    print("=" * 85)

    overall_summary = {}
    cum_total_prompt_tokens = 0
    cum_total_completion_tokens = 0
    cum_total_cost_usd = 0.0

    for dom_idx, domain in enumerate(domains, start=1):
        test_path = os.path.join(args.test_dir, f"{domain}.json")
        if not os.path.exists(test_path):
            print(f"\n⚠️ [{dom_idx}/{len(domains)}] Test file not found for domain '{domain}': {test_path}")
            continue

        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        exemplar = find_exemplar(test_data)
        if not exemplar:
            print(f"\n⚠️ [{dom_idx}/{len(domains)}] No exemplar with valid plan found for domain '{domain}'")
            continue

        q_ex = f"Context:\n{exemplar.get('context','')}\n\nQuestion:\n{exemplar.get('inputs','')}"
        s_ex = str(exemplar.get("sample_plan"))

        print(f"\n🔨 [{dom_idx}/{len(domains)}] Building SCOPE Solver for Domain: {domain} (Exemplar ID: {exemplar.get('id')})...")
        
        # Reset token counter before build
        token_usage["prompt_tokens"] = 0
        token_usage["completion_tokens"] = 0
        token_usage["api_calls"] = 0
        
        build_start = time.time()
        try:
            builder = SCOPESolverBuilder(domain, q_ex, s_ex)
            config = builder.build()
            solver = StandaloneSCOPESolver(config)
            build_time = time.time() - build_start
        except Exception as e:
            print(f"❌ Failed to build SCOPE solver for {domain}: {e}")
            continue

        build_prompt_tokens = config.get("build_prompt_tokens", token_usage["prompt_tokens"])
        build_completion_tokens = config.get("build_completion_tokens", token_usage["completion_tokens"])
        build_cost_usd = calculate_cost(build_prompt_tokens, build_completion_tokens)

        print(f"   ✓ Solver built in {build_time:.2f}s | Build Cost: ${build_cost_usd:.4f} ({build_prompt_tokens + build_completion_tokens} tokens)")

        if args.limit:
            test_data = test_data[:args.limit]

        n_inst = len(test_data)
        print(f"📌 Evaluating {n_inst} instances for Domain: {domain}...")
        print("-" * 85)

        domain_results = []
        correct_decisions = 0
        val_valid_plans = 0
        gt_yes_count = 0

        query_prompt_tokens = 0
        query_completion_tokens = 0

        for idx, item in enumerate(test_data, start=1):
            ex_id = item.get("id")
            context = item.get("context", "")
            inputs = item.get("inputs", "")
            gt_ans = str(item.get("answer", "no")).strip().lower()

            start_t = time.time()
            try:
                res_dict = solver.solve(context, inputs)
                pred_ans = "yes" if _normalize_yes(res_dict.get("ans")) else "no"
                pred_plan = res_dict.get("plan", [])
                reason = res_dict.get("reason", "")
            except Exception as e:
                pred_ans = "no"
                pred_plan = []
                reason = f"Execution error: {e}"
            elapsed = time.time() - start_t

            is_gt_yes = (gt_ans == "yes")
            is_pred_yes = (pred_ans == "yes")

            if is_gt_yes:
                gt_yes_count += 1
            if gt_ans == pred_ans:
                correct_decisions += 1

            val_valid = False
            if args.validate and is_pred_yes and pred_plan:
                val_valid = validate_plan_with_val(item.get("PDDL_domain", ""), item.get("PDDL_problem", ""), pred_plan)
                if val_valid and is_gt_yes:
                    val_valid_plans += 1

            val_str = "VALID" if val_valid else ("INVALID" if is_pred_yes else "N/A")
            print(f" [{idx:3d}/{n_inst:3d}] ID: {ex_id:<35s} | GT: {gt_ans:<3s} | Pred: {pred_ans:<3s} | VAL: {val_str:<7s} | Time: {elapsed:.2f}s")

            domain_results.append({
                "id": ex_id,
                "plan_existence": pred_ans,
                "predicted_plan": pred_plan,
                "reason": reason,
                "gt_answer": gt_ans,
                "val_valid": val_valid
            })

        domain_total_prompt_tokens = build_prompt_tokens + query_prompt_tokens
        domain_total_completion_tokens = build_completion_tokens + query_completion_tokens
        domain_total_cost_usd = build_cost_usd + calculate_cost(query_prompt_tokens, query_completion_tokens)

        out_file = os.path.join(args.output_dir, f"{domain}_scope_test_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "domain": domain,
                "split": "test",
                "instances": len(test_data),
                "build_prompt_tokens": build_prompt_tokens,
                "build_completion_tokens": build_completion_tokens,
                "build_cost_usd": build_cost_usd,
                "domain_total_prompt_tokens": domain_total_prompt_tokens,
                "domain_total_completion_tokens": domain_total_completion_tokens,
                "domain_total_cost_usd": domain_total_cost_usd,
                "results": domain_results
            }, f, indent=2)

        dec_acc = (correct_decisions / n_inst * 100) if n_inst > 0 else 0.0
        val_acc = (val_valid_plans / gt_yes_count * 100) if gt_yes_count > 0 else 0.0

        cum_total_prompt_tokens += domain_total_prompt_tokens
        cum_total_completion_tokens += domain_total_completion_tokens
        cum_total_cost_usd += domain_total_cost_usd

        print("-" * 85)
        print(f"  Summary {domain}: Dec Acc = {dec_acc:.2f}% | VAL Plan Acc = {val_acc:.2f}% | Domain Cost = ${domain_total_cost_usd:.4f}")

        overall_summary[domain] = {
            "instances": len(test_data),
            "correct_decisions": correct_decisions,
            "decision_accuracy": dec_acc,
            "val_valid_plans": val_valid_plans,
            "val_plan_accuracy": val_acc,
            "build_cost_usd": build_cost_usd,
            "domain_cost_usd": domain_total_cost_usd
        }

    summary_file = os.path.join(args.output_dir, "scope_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_prompt_tokens": cum_total_prompt_tokens,
            "total_completion_tokens": cum_total_completion_tokens,
            "total_tokens": cum_total_prompt_tokens + cum_total_completion_tokens,
            "total_cost_usd": cum_total_cost_usd,
            "domains": overall_summary
        }, f, indent=2)

    print("\n" + "=" * 85)
    print(f" 🎉 SCOPE EVALUATION COMPLETED!")
    print(f" 📊 Total Prompt Tokens     : {cum_total_prompt_tokens:,}")
    print(f" 📊 Total Completion Tokens : {cum_total_completion_tokens:,}")
    print(f" 💵 Total Cumulative Cost   : ${cum_total_cost_usd:.4f} USD")
    print(f" 💾 Summary File Saved To   : {summary_file}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
