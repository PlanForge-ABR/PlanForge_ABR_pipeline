#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) Baseline Evaluator for ACPBench.
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

from openai import OpenAI
from evaluation.action_val import fix_pddl_domain_typing

VAL_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VAL", "build", "macos64", "Release", "bin", "Validate")

# Setup OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
if base_url:
    client = OpenAI(api_key=openai_api_key, base_url=base_url)
else:
    client = OpenAI(api_key=openai_api_key)

# Domain Descriptions
DOMAIN_DESCRIPTIONS = {
    "blocksworld": "This is a blocksworld domain where blocks can be placed on top of each other or on the table. A robotic arm can pick up, put down, stack, and unstack blocks.",
    "ferry": "This is a ferry transportation domain where a ferry moves cars between different locations.",
    "logistics": "This is a logistics domain involving packages, trucks, airplanes, and locations.",
    "grippers": "This is a grippers domain where robots with grippers move balls between rooms.",
    "rovers": "This is a planetary rover domain where rovers navigate terrain to collect samples and data.",
    "visitall": "This is a visit-all domain where an agent must visit every location in a grid.",
    "grid": "This is a grid navigation domain where an agent moves on a rectangular grid.",
    "floortile": "This is a floor tiling domain where robots paint tiles on a floor in specific colors.",
    "alfworld": "This is an embodied household-task domain where an agent interacts with everyday objects in rooms.",
    "depot": "This is a supply-depot management domain with pallets, crates, trucks, and hoists.",
    "goldminer": "This is a mining domain where a miner navigates tunnels to extract gold.",
    "satellite": "This is a satellite-imaging domain where satellites capture images using instruments.",
    "swap": "This is a swapping domain where agents exchange the positions of objects across locations.",
    "frogs_jumping": "This is a puzzle domain where frogs jump or slide across lily pads.",
    "hanoi": "This is a Tower of Hanoi domain where disks are moved between pegs obeying size constraints."
}

def calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "gpt-5.4") -> float:
    """Calculate USD cost based on token counts (GPT-5.4 rate: $2.50/1M input, $15.00/1M output)."""
    return (prompt_tokens * 2.50 + completion_tokens * 15.00) / 1000000.0

def validate_plan_with_val(pddl_domain: str, pddl_problem: str, plan_actions: List[str]) -> bool:
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

def run_cot_on_example(domain: str, example: Dict[str, Any], model_name: str = "gpt-5.4") -> Dict[str, Any]:
    """Query model using Chain-of-Thought prompting and track token usage & cost."""
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "")
    context = example.get("context", "")
    inputs = example.get("inputs", "")
    
    prompt = f"""{domain_desc}

{context}

Question:
{inputs}

Instructions:
1. Think step-by-step about whether it is possible to transition to the requested target state.
2. If it IS possible, list the exact sequence of action steps required (one per line, e.g. "pick-up block_1").
3. End your response with a line stating exactly:
Final Answer: yes
OR
Final Answer: no
"""
    
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert AI planning assistant. Solve the planning query using chain-of-thought reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
    except Exception as e:
        logging.error(f"API Call failed for example {example.get('id')}: {e}")
        return {
            "id": example.get("id"),
            "plan_existence": "no",
            "predicted_plan": [],
            "reasoning": f"API Error: {e}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        }

    # Extract final answer
    plan_existence = "no"
    if "final answer: yes" in content.lower():
        plan_existence = "yes"
    elif "final answer: no" in content.lower():
        plan_existence = "no"

    # Extract plan steps if yes
    predicted_plan = []
    if plan_existence == "yes":
        lines = content.split("\n")
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("- ") or line_str.startswith("* ") or (len(line_str) > 2 and line_str[0].isdigit() and line_str[1] in ". "):
                line_str = line_str.lstrip("- *0123456789. ").strip()
            if line_str and not line_str.lower().startswith("final answer") and not line_str.lower().startswith("step"):
                predicted_plan.append(line_str)

    cost_usd = calculate_cost(prompt_tokens, completion_tokens, model_name=model_name)

    return {
        "id": example.get("id"),
        "plan_existence": plan_existence,
        "predicted_plan": predicted_plan,
        "reasoning": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd
    }

def main():
    parser = argparse.ArgumentParser(description="Run Chain-of-Thought (CoT) on ACPBench dataset.")
    parser.add_argument("--domain", default="all", help="Domain to evaluate (e.g. blocksworld, ferry) or 'all'.")
    parser.add_argument("--test_dir", default="data/ACPBench_dataset_final/test", help="Path to ACPBench test dataset folder.")
    parser.add_argument("--output_dir", default="new_results/cot", help="Directory to save output results.")
    parser.add_argument("--model", default="gpt-5.4", help="Model name (e.g. gpt-5.4, gpt-5.1).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test examples per domain.")
    parser.add_argument("--validate", action="store_true", default=True, help="Run VAL validation on generated plans.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.domain.lower() == "all":
        domain_files = sorted(glob.glob(os.path.join(args.test_dir, "*.json")))
        domains = [os.path.basename(f).replace(".json", "") for f in domain_files]
    else:
        domains = [d.strip() for d in args.domain.split(",")]

    print(f"\n" + "=" * 85)
    print(f" 🚀 STARTING CoT EVALUATION ON {len(domains)} DOMAIN(S) | MODEL: {args.model}")
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

        if args.limit:
            test_data = test_data[:args.limit]

        n_inst = len(test_data)
        print(f"\n📌 [{dom_idx}/{len(domains)}] Domain: {domain:<15s} | {n_inst} test instances")
        print("-" * 85)

        domain_results = []
        correct_decisions = 0
        val_valid_plans = 0
        gt_yes_count = 0
        domain_prompt_tokens = 0
        domain_completion_tokens = 0
        domain_cost_usd = 0.0

        for idx, item in enumerate(test_data, start=1):
            start_t = time.time()
            res = run_cot_on_example(domain, item, model_name=args.model)
            elapsed = time.time() - start_t

            gt_ans = str(item.get("answer", "no")).strip().lower()
            pred_ans = res["plan_existence"].strip().lower()
            
            is_gt_yes = (gt_ans == "yes")
            is_pred_yes = (pred_ans == "yes")
            
            if is_gt_yes:
                gt_yes_count += 1
            if gt_ans == pred_ans:
                correct_decisions += 1

            val_valid = False
            if args.validate and is_pred_yes and res["predicted_plan"]:
                val_valid = validate_plan_with_val(item.get("PDDL_domain", ""), item.get("PDDL_problem", ""), res["predicted_plan"])
                if val_valid and is_gt_yes:
                    val_valid_plans += 1

            domain_prompt_tokens += res["prompt_tokens"]
            domain_completion_tokens += res["completion_tokens"]
            domain_cost_usd += res["cost_usd"]

            val_str = "VALID" if val_valid else ("INVALID" if is_pred_yes else "N/A")
            print(f" [{idx:3d}/{n_inst:3d}] ID: {item.get('id'):<35s} | GT: {gt_ans:<3s} | Pred: {pred_ans:<3s} | VAL: {val_str:<7s} | Tokens: {res['total_tokens']:4d} | Cost: ${res['cost_usd']:.5f} | Time: {elapsed:.2f}s")

            res["gt_answer"] = gt_ans
            res["val_valid"] = val_valid
            domain_results.append(res)

        out_file = os.path.join(args.output_dir, f"{domain}_cot_test_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "domain": domain,
                "split": "test",
                "instances": len(test_data),
                "domain_prompt_tokens": domain_prompt_tokens,
                "domain_completion_tokens": domain_completion_tokens,
                "domain_total_tokens": domain_prompt_tokens + domain_completion_tokens,
                "domain_cost_usd": domain_cost_usd,
                "results": domain_results
            }, f, indent=2)

        dec_acc = (correct_decisions / n_inst * 100) if n_inst > 0 else 0.0
        val_acc = (val_valid_plans / gt_yes_count * 100) if gt_yes_count > 0 else 0.0

        cum_total_prompt_tokens += domain_prompt_tokens
        cum_total_completion_tokens += domain_completion_tokens
        cum_total_cost_usd += domain_cost_usd

        print("-" * 85)
        print(f"  Summary {domain}: Dec Acc = {dec_acc:.2f}% | VAL Plan Acc = {val_acc:.2f}% | Domain Cost = ${domain_cost_usd:.4f}")

        overall_summary[domain] = {
            "instances": len(test_data),
            "correct_decisions": correct_decisions,
            "decision_accuracy": dec_acc,
            "val_valid_plans": val_valid_plans,
            "val_plan_accuracy": val_acc,
            "prompt_tokens": domain_prompt_tokens,
            "completion_tokens": domain_completion_tokens,
            "total_tokens": domain_prompt_tokens + domain_completion_tokens,
            "cost_usd": domain_cost_usd
        }

    summary_file = os.path.join(args.output_dir, "cot_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_prompt_tokens": cum_total_prompt_tokens,
            "total_completion_tokens": cum_total_completion_tokens,
            "total_tokens": cum_total_prompt_tokens + cum_total_completion_tokens,
            "total_cost_usd": cum_total_cost_usd,
            "domains": overall_summary
        }, f, indent=2)

    print("\n" + "=" * 85)
    print(f" 🎉 CoT EVALUATION COMPLETED!")
    print(f" 📊 Total Prompt Tokens     : {cum_total_prompt_tokens:,}")
    print(f" 📊 Total Completion Tokens : {cum_total_completion_tokens:,}")
    print(f" 💵 Total Cumulative Cost   : ${cum_total_cost_usd:.4f} USD")
    print(f" 💾 Summary File Saved To   : {summary_file}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
