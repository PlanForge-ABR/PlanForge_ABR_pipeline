#!/usr/bin/env python3
import os
import re
import sys
import time
import json
import logging
import argparse
import concurrent.futures
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Resolve absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Initialize logging
os.makedirs(os.path.join(project_root, "baseline_results/structured_sat"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

DOMAIN_SUBDIRS = {
    "3-sat_balanced": "train/sat",
    "automor_balanced": "mnt/brain7/scratch/violetfy/k-isomo/train/unsat",
    "ca_balanced": "ca/train/sat",
    "k-clique_balanced": "mnt/brain7/scratch/violetfy/k-clique/train/unsat",
    "k-color_balanced": "mnt/brain7/scratch/violetfy/k-color/train/unsat",
    "k-domset_balanced": "mnt/brain7/scratch/violetfy/k-domset/train/unsat",
    "k-vercov_balanced": "train/sat",
    "ps_balanced": "train/sat",
    "sr_balanced": "train/sat"
}

# Pricing for gpt-5.4
def calculate_cost(prompt_tokens, completion_tokens):
    return (prompt_tokens * 2.50 + completion_tokens * 15.00) / 1000000.0

def parse_cnf_file(file_path):
    with open(file_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    
    variables = 0
    clauses = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("c"):
            continue
        if line.startswith("p cnf"):
            parts = line.split()
            if len(parts) >= 4:
                variables = int(parts[2])
            continue
        
        # Parse clause literals (ended by 0)
        parts = [int(x) for x in line.split()]
        if parts:
            if parts[-1] == 0:
                parts = parts[:-1]
            if parts:
                clauses.append(parts)
                
    return {
        "variables": variables,
        "clauses": clauses
    }

def check_sat_solution_satisfies_clauses(assignment, clauses):
    if not assignment:
        return False
    if isinstance(assignment, dict):
        assignment_dict = {int(k): bool(v) for k, v in assignment.items()}
    else:
        assignment_dict = {}
        for lit in assignment:
            val = int(lit)
            assignment_dict[abs(val)] = (val > 0)
            
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            sign = (lit > 0)
            if var in assignment_dict:
                if assignment_dict[var] == sign:
                    clause_satisfied = True
                    break
        if not clause_satisfied:
            return False
    return True

def extract_json_robust(text: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    match_braces = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_braces:
        try:
            return json.loads(match_braces.group(1).strip())
        except Exception:
            pass
    return {}

def normalize_answer(ans: str) -> str:
    ans_clean = str(ans).strip().lower()
    if "unsat" in ans_clean or ans_clean == "no":
        return "unsat"
    if "sat" in ans_clean or ans_clean == "yes":
        return "sat"
    return ans_clean

def query_llm(client: OpenAI, prompt: str, system_prompt: str, model_name: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        
        return {
            "content": response.choices[0].message.content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": calculate_cost(prompt_tokens, completion_tokens),
            "api_calls": 1
        }
    except Exception as e:
        logging.error(f"Error querying OpenAI model: {e}")
        raise e

def process_example(args_tuple):
    idx, filename, domain, domain_dir, gt_answer, cnf_data, mode, model_name, client, prompt_template, system_prompt = args_tuple
    example_id = filename.replace(".cnf", "")
    file_path = os.path.join(domain_dir, filename)
    clauses_str = "\n".join(" ".join(map(str, c)) + " 0" for c in cnf_data["clauses"])
    prompt = prompt_template.format(variables=cnf_data["variables"], clauses_str=clauses_str)

    logging.info(f"🔄 [{domain}] starting example_id={example_id} (vars={cnf_data['variables']}, clauses={len(cnf_data['clauses'])})")
    
    start_time = time.time()
    try:
        solve_res = query_llm(client, prompt, system_prompt, model_name)
        time_taken = time.time() - start_time
        
        response_text = solve_res["content"]
        parsed_json = extract_json_robust(response_text)
        
        final_answer = normalize_answer(parsed_json.get("final_answer", ""))
        assignment = parsed_json.get("assignment", {})
        
        is_correct = False
        is_valid = False
        validation_reason = None
        
        if final_answer == gt_answer:
            is_correct = True
            is_valid = True
            
        if final_answer == "sat":
            if assignment:
                assignment_valid = check_sat_solution_satisfies_clauses(assignment, cnf_data["clauses"])
                if not assignment_valid:
                    is_valid = False
                    validation_reason = "Assignment does not satisfy the CNF clauses."
            else:
                is_valid = False
                validation_reason = "Predicted 'sat' but no assignment dictionary was returned."
                
        return {
            "example_id": example_id,
            "file_path": file_path,
            "parsed_query": cnf_data,
            "prompt": prompt,
            "response": response_text,
            "plan": assignment,
            "final_answer": final_answer,
            "ground_truth_answer": gt_answer,
            "validation_reason": validation_reason,
            "time_taken": time_taken,
            "correct": is_correct,
            "valid": is_valid,
            "prompt_tokens": solve_res["prompt_tokens"],
            "completion_tokens": solve_res["completion_tokens"],
            "api_calls": solve_res["api_calls"],
            "cost": solve_res["cost"]
        }
    except Exception as e:
        logging.error(f"Failed to process example {example_id} in {domain}: {e}")
        return {
            "example_id": example_id,
            "file_path": file_path,
            "parsed_query": cnf_data,
            "error": str(e),
            "correct": False,
            "valid": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "api_calls": 0,
            "cost": 0.0
        }

def save_summary(summary_path, summary_results, mode, model_name):
    existing_summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                existing_summary = json.load(fh)
        except Exception:
            pass

    merged_domains = existing_summary.get("domains", {})
    merged_domains.update(summary_results)

    overall_total = sum(d["total"] for d in merged_domains.values())
    overall_correct = sum(d["exact_match_correct"] for d in merged_domains.values())
    overall_valid = sum(d["success_rate_correct"] for d in merged_domains.values())

    overall_em_acc = 100.0 * overall_correct / overall_total if overall_total > 0 else 0.0
    overall_sr_acc = 100.0 * overall_valid / overall_total if overall_total > 0 else 0.0

    overall_prompt = sum(d["query_prompt_tokens"] for d in merged_domains.values())
    overall_comp = sum(d["query_completion_tokens"] for d in merged_domains.values())
    overall_api_calls = sum(d.get("query_api_calls", 0) for d in merged_domains.values())
    overall_cost = sum(d["query_cost_usd"] for d in merged_domains.values())

    summary_output = {
        "metadata": {
            "model": model_name,
            "mode": mode,
            "overall_exact_match_accuracy": f"{overall_em_acc:.2f}%",
            "overall_success_rate": f"{overall_sr_acc:.2f}%",
            "overall_total": overall_total,
            "overall_correct_em": overall_correct,
            "overall_correct_sr": overall_valid,
            "query_prompt_tokens": overall_prompt,
            "query_completion_tokens": overall_comp,
            "query_api_calls": overall_api_calls,
            "query_cost_usd": overall_cost,
            "total_prompt_tokens": overall_prompt,
            "total_completion_tokens": overall_comp,
            "total_api_calls": overall_api_calls,
            "total_cost_usd": overall_cost
        },
        "domains": merged_domains
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_output, fh, indent=2)
    return summary_output

def main():
    parser = argparse.ArgumentParser(description="Run zero-shot and CoT baselines on StructuredSAT in parallel.")
    parser.add_argument("--mode", type=str, required=True, choices=["zero_shot", "cot"], help="Run mode")
    parser.add_argument("--limit", type=int, default=150, help="Limit number of examples per domain")
    parser.add_argument("--domain", type=str, default="all", help="Comma separated domains or 'all'")
    parser.add_argument("--model-name", type=str, default="gpt-5.4", help="Model name to call")
    parser.add_argument("--output_dir", type=str, default="baseline_results/structured_sat", help="Output directory")
    parser.add_argument("--max_workers", type=int, default=20, help="Number of parallel threads for calling LLM")
    args = parser.parse_args()

    # Initialize OpenAI client
    openai_api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL')
    if base_url:
        client = OpenAI(api_key=openai_api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=openai_api_key)

    # Determine domains to run
    if args.domain and args.domain.lower() != "all":
        domains = [d.strip() for d in args.domain.split(",")]
    else:
        domains = sorted(list(DOMAIN_SUBDIRS.keys()))

    logging.info(f"Starting {args.mode} baseline on {len(domains)} StructuredSAT domains: {', '.join(domains)} with max_workers={args.max_workers}")
    
    summary_results = {}
    summary_path = os.path.join(project_root, args.output_dir, f"{args.mode}_accuracy_summary.json")

    # Mode prompts
    if args.mode == "zero_shot":
        system_prompt = "You are an expert Boolean Satisfiability (SAT) solver. Your task is to solve the Boolean Satisfiability problem given in Conjunctive Normal Form (CNF)."
        prompt_template = """Solve the Boolean Satisfiability problem.
Number of variables: {variables}
Clauses (using standard DIMACS CNF representation, where negative numbers represent negated variables, e.g. -1 means NOT x1):
{clauses_str}

Is this formula satisfiable? Answer with either "sat" or "unsat".
If it is satisfiable ("sat"), you MUST also provide a valid satisfying assignment of all variables.
Format your final response in JSON with the following structure:
{{
  "final_answer": "sat" or "unsat",
  "assignment": {{
    "1": true,
    "2": false,
    ...
  }}
}}
Do not write anything else outside the JSON block."""
    else: # cot
        system_prompt = "You are an expert Boolean Satisfiability (SAT) solver. Your task is to solve the Boolean Satisfiability problem given in Conjunctive Normal Form (CNF) by reasoning step-by-step."
        prompt_template = """Solve the Boolean Satisfiability problem.
Number of variables: {variables}
Clauses (using standard DIMACS CNF representation, where negative numbers represent negated variables, e.g. -1 means NOT x1):
{clauses_str}

Reason step-by-step to find if there is a satisfying assignment.
Then, decide if the formula is satisfiable.
Answer with either "sat" or "unsat".
If it is satisfiable ("sat"), you must provide a valid satisfying assignment of all variables.

Format your final response with two parts:
1. Reasoning: Your step-by-step reasoning.
2. Final Answer: Format this part as a JSON block with the structure:
```json
{{
  "final_answer": "sat" or "unsat",
  "assignment": {{
    "1": true,
    "2": false,
    ...
  }}
}}
```"""

    for domain in domains:
        logging.info(f"\n===== Domain: {domain} =====")
        sub_path = DOMAIN_SUBDIRS.get(domain)
        if not sub_path:
            logging.error(f"Unknown domain mapping for {domain}")
            continue

        domain_dir = os.path.join(project_root, "data/StructuredSAT", domain, sub_path)
        if not os.path.exists(domain_dir):
            logging.error(f"Domain directory not found: {domain_dir}")
            continue

        cnf_files = sorted([f for f in os.listdir(domain_dir) if f.endswith(".cnf")])
        if not cnf_files:
            logging.error(f"No CNF files found in {domain_dir}")
            continue

        if "unsat" in sub_path:
            gt_answer = "unsat"
        else:
            gt_answer = "sat"

        logging.info(f"Domain '{domain}' has {len(cnf_files)} instances. Ground Truth: {gt_answer}")

        if args.limit and len(cnf_files) > args.limit:
            cnf_files = cnf_files[:args.limit]
            logging.info(f"Limiting execution to {args.limit} examples.")

        domain_output_dir = os.path.join(project_root, args.output_dir, f"output_full_{args.mode}_{args.model_name}")
        os.makedirs(domain_output_dir, exist_ok=True)
        domain_output_path = os.path.join(domain_output_dir, f"{domain}_{args.mode}.json")

        # Prep all tasks for ThreadPoolExecutor
        tasks = []
        for idx, filename in enumerate(cnf_files, start=1):
            file_path = os.path.join(domain_dir, filename)
            cnf_data = parse_cnf_file(file_path)
            tasks.append((idx, filename, domain, domain_dir, gt_answer, cnf_data, args.mode, args.model_name, client, prompt_template, system_prompt))

        results = []
        domain_correct = 0
        domain_valid = 0
        total_query_prompt_tokens = 0
        total_query_completion_tokens = 0
        total_query_cost = 0.0
        total_query_api_calls = 0

        # Execute tasks in parallel
        completed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {executor.submit(process_example, t): t for t in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                res = future.result()
                results.append(res)
                completed_count += 1
                
                example_id = res.get("example_id")
                logging.info(f"✅ [{domain}] finished example_id={example_id} ({completed_count}/{len(tasks)} completed)")
                
                if res.get("correct"):
                    domain_correct += 1
                if res.get("valid"):
                    domain_valid += 1
                total_query_prompt_tokens += res.get("prompt_tokens", 0)
                total_query_completion_tokens += res.get("completion_tokens", 0)
                total_query_cost += res.get("cost", 0.0)
                total_query_api_calls += res.get("api_calls", 0)

        # Sort results by example_id to keep output stable and clean
        results.sort(key=lambda x: x.get("example_id", ""))

        with open(domain_output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"💾 Test results saved to {domain_output_path}")

        domain_total = len(results)
        if domain_total > 0:
            em_acc = 100.0 * domain_correct / domain_total
            sr_acc = 100.0 * domain_valid / domain_total
            
            summary_results[domain] = {
                "total": domain_total,
                "exact_match_correct": domain_correct,
                "exact_match_accuracy": f"{em_acc:.2f}%",
                "success_rate_correct": domain_valid,
                "success_rate": f"{sr_acc:.2f}%",
                "build_prompt_tokens": 0,
                "build_completion_tokens": 0,
                "build_api_calls": 0,
                "build_cost_usd": 0.0,
                "query_prompt_tokens": total_query_prompt_tokens,
                "query_completion_tokens": total_query_completion_tokens,
                "query_api_calls": total_query_api_calls,
                "query_cost_usd": total_query_cost,
                "total_prompt_tokens": total_query_prompt_tokens,
                "total_completion_tokens": total_query_completion_tokens,
                "total_api_calls": total_query_api_calls,
                "total_cost_usd": total_query_cost
            }

            save_summary(summary_path, summary_results, args.mode, args.model_name)
            logging.info(f"💾 Summary updated → {summary_path}")

    # Final summary log
    final = save_summary(summary_path, summary_results, args.mode, args.model_name)
    meta = final["metadata"]
    logging.info(f"Final summary saved to {summary_path}")
    logging.info(f"🎯 Mode: {args.mode} | Overall exact match accuracy: {meta['overall_exact_match_accuracy']} | Overall success rate: {meta['overall_success_rate']}")

if __name__ == "__main__":
    main()
