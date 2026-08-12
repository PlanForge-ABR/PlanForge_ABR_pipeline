import os
import sys
import time
import json
import argparse
import logging
import re

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from scope.workflow_natural_plan import SCOPESolverBuilder, StandaloneSCOPESolver, calculate_cost, validate_plan_llm, token_usage
except ImportError:
    from workflow_natural_plan import SCOPESolverBuilder, StandaloneSCOPESolver, calculate_cost, validate_plan_llm, token_usage

# Setup logging
os.makedirs("scope/results/natural_plan", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scope/results/natural_plan/run_natural_plan.log"),
        logging.StreamHandler()
    ]
)

def discover_domains(test_root):
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test root directory not found: {test_root}")
    
    domains = []
    for fname in os.listdir(test_root):
        if fname.endswith(".json"):
            domain = fname[:-5]  # Remove .json
            if domain:
                domains.append(domain)
    return sorted(list(set(domains)))

def compare_answers(pred, gt):
    if pred is None or gt is None:
        return False
    # If both are lists
    if isinstance(pred, list) and isinstance(gt, list):
        return [str(x).strip().lower() for x in pred] == [str(x).strip().lower() for x in gt]
    
    # Otherwise normalize to clean strings (ignore all duplicate whitespace and newlines)
    if isinstance(pred, list):
        pred_str = "\n".join(str(x).strip().lower() for x in pred)
    else:
        pred_str = str(pred).strip().lower()
        
    if isinstance(gt, list):
        gt_str = "\n".join(str(x).strip().lower() for x in gt)
    else:
        gt_str = str(gt).strip().lower()
        
    pred_str_clean = re.sub(r'\s+', ' ', pred_str).strip()
    gt_str_clean = re.sub(r'\s+', ' ', gt_str).strip()
    
    return pred_str_clean == gt_str_clean

def save_summary(summary_path, summary_results):
    """Load existing summary, merge new domain results, and write back."""
    existing_summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                existing_summary = json.load(fh)
        except Exception:
            pass

    merged_domains = existing_summary.get("domains", {})
    merged_domains.update(summary_results)   # overwrite domains we just ran

    overall_total   = sum(d["total"]                 for d in merged_domains.values())
    overall_correct = sum(d["exact_match_correct"]   for d in merged_domains.values())
    overall_valid   = sum(d["success_rate_correct"]  for d in merged_domains.values())

    overall_em_acc = 100.0 * overall_correct / overall_total if overall_total > 0 else 0.0
    overall_sr_acc = 100.0 * overall_valid   / overall_total if overall_total > 0 else 0.0

    overall_build_prompt = sum(d["build_prompt_tokens"]    for d in merged_domains.values())
    overall_build_comp   = sum(d["build_completion_tokens"] for d in merged_domains.values())
    overall_build_api_calls = sum(d.get("build_api_calls", 0) for d in merged_domains.values())
    overall_build_cost   = sum(d["build_cost_usd"]         for d in merged_domains.values())
    overall_query_prompt = sum(d["query_prompt_tokens"]    for d in merged_domains.values())
    overall_query_comp   = sum(d["query_completion_tokens"] for d in merged_domains.values())
    overall_query_api_calls = sum(d.get("query_api_calls", 0) for d in merged_domains.values())
    overall_query_cost   = sum(d["query_cost_usd"]         for d in merged_domains.values())
    overall_total_prompt = overall_build_prompt + overall_query_prompt
    overall_total_comp   = overall_build_comp   + overall_query_comp
    overall_total_api_calls = overall_build_api_calls + overall_query_api_calls
    overall_total_cost   = overall_build_cost   + overall_query_cost

    summary_output = {
        "metadata": {
            "model": "gpt-5.4",
            "overall_exact_match_accuracy": f"{overall_em_acc:.2f}%",
            "overall_success_rate":         f"{overall_sr_acc:.2f}%",
            "overall_total":           overall_total,
            "overall_correct_em":      overall_correct,
            "overall_correct_sr":      overall_valid,
            "build_prompt_tokens":     overall_build_prompt,
            "build_completion_tokens": overall_build_comp,
            "build_api_calls":         overall_build_api_calls,
            "build_cost_usd":          overall_build_cost,
            "query_prompt_tokens":     overall_query_prompt,
            "query_completion_tokens": overall_query_comp,
            "query_api_calls":         overall_query_api_calls,
            "query_cost_usd":          overall_query_cost,
            "total_prompt_tokens":     overall_total_prompt,
            "total_completion_tokens": overall_total_comp,
            "total_api_calls":         overall_total_api_calls,
            "total_cost_usd":          overall_total_cost
        },
        "domains": merged_domains
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_output, fh, indent=2)
    return summary_output


def main():
    parser = argparse.ArgumentParser(description="Run Standalone SCOPE on Natural Plan datasets.")
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain name (e.g. calendar_scheduling, meeting_planning, trip_planning). Use 'all' (default)."
    )
    parser.add_argument(
        "--test_root",
        default="data/natural-plan",
        help="Root directory containing natural plan JSON files."
    )
    parser.add_argument(
        "--output_dir",
        default="scope/results/natural_plan",
        help="Directory to save the outputs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to run per domain (default: run all)."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip domains that already have summary results in accuracy_summary.json or output files."
    )
    
    args = parser.parse_args()
    
    # 1. Discover domains
    if args.domain and args.domain.lower() != "all":
        domains = [d.strip() for d in args.domain.split(",")]
    else:
        domains = discover_domains(args.test_root)
        
    if args.skip_existing:
        summary_path = os.path.join(args.output_dir, "accuracy_summary.json")
        existing_domains = []
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as fh:
                    existing_summary = json.load(fh)
                    existing_domains = list(existing_summary.get("domains", {}).keys())
            except Exception:
                pass
                
        domains_to_run = []
        for d in domains:
            out_file = os.path.join(args.output_dir, "output_full_gpt-5.4", f"{d}_scope.json")
            if d in existing_domains or os.path.exists(out_file):
                logging.info(f"⏭️ Skipping already completed domain: {d}")
            else:
                domains_to_run.append(d)
        domains = domains_to_run
        
    logging.info(f"🚀 Starting Standalone SCOPE execution on {len(domains)} Natural Plan domains: {', '.join(domains)}")
    
    overall_total = 0
    overall_correct = 0
    overall_valid = 0
    summary_results = {}
    
    for domain in domains:
        logging.info(f"\n===== Domain: {domain} =====")
        
        # Load JSON data
        test_path = os.path.join(args.test_root, f"{domain}.json")
        if not os.path.exists(test_path):
            logging.error(f"Test file not found: {test_path}")
            continue
            
        with open(test_path, "r", encoding="utf-8") as fh:
            raw_data = json.load(fh)
            
        # Convert dictionary to sorted list of examples
        test_data = []
        for key in sorted(raw_data.keys(), key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x):
            val = raw_data[key]
            val["id"] = key
            test_data.append(val)
            
        if not test_data:
            logging.error(f"No examples found in dataset {test_path}!")
            continue
            
        # Select first example as exemplar
        exemplar_example = test_data[0]
        q_ex = exemplar_example.get("prompt_0shot")
        s_ex = exemplar_example.get("golden_plan")
        
        logging.info(f"Using exemplar ID: {exemplar_example.get('id')}")
        
        # Build solver using the exemplar
        try:
            builder = SCOPESolverBuilder(domain, q_ex, s_ex)
            config = builder.build()
            solver = StandaloneSCOPESolver(config)
        except Exception as e:
            import traceback
            logging.error(f"Failed to build solver for domain {domain}: {e}\n{traceback.format_exc()}")
            continue
            
        build_prompt_tokens = config.get("build_prompt_tokens", 0)
        build_completion_tokens = config.get("build_completion_tokens", 0)
        build_api_calls = config.get("build_api_calls", 0)
        build_cost = config.get("build_cost", 0.0)
            
        # Save solver code checkpoints for visibility
        solver_code_dir = os.path.join(args.output_dir, "solver_codes")
        os.makedirs(solver_code_dir, exist_ok=True)
        with open(os.path.join(solver_code_dir, f"{domain}_combinations.py"), "w") as f:
            f.write(config["combinations_func_code"])
        with open(os.path.join(solver_code_dir, f"{domain}_plan.py"), "w") as f:
            f.write(config["plan_func_code"])
        with open(os.path.join(solver_code_dir, f"{domain}_deliver.py"), "w") as f:
            f.write(config["deliver_func_code"])
            
        if args.limit and len(test_data) > args.limit:
            test_data = test_data[:args.limit]
            logging.info(f"⚠️ Limiting execution to {args.limit} examples.")
            
        logging.info(f"📥 Loaded {len(test_data)} test examples from {test_path}")
        
        domain_output_dir = os.path.join(args.output_dir, "output_full_gpt-5.4")
        os.makedirs(domain_output_dir, exist_ok=True)
        domain_output_path = os.path.join(domain_output_dir, f"{domain}_scope.json")
        
        results = []
        domain_total = 0
        domain_correct = 0  # EM correct
        domain_valid = 0    # success rate (validity)
        total_query_prompt_tokens = 0
        total_query_completion_tokens = 0
        total_query_cost = 0.0
        total_query_api_calls = 0
        
        for idx, example in enumerate(test_data, start=1):
            example_id = example.get("id")
            query = example.get("prompt_0shot")
            gt_answer = example.get("golden_plan")
            
            logging.info(f"🔄 [{idx}/{len(test_data)}] example_id={example_id}")
            
            start_time = time.time()
            solve_res = solver.solve(query)
            time_taken = time.time() - start_time
            
            final_answer = solve_res["final_answer"]
            
            # Accumulate query stats
            total_query_prompt_tokens += solve_res.get("prompt_tokens", 0)
            total_query_completion_tokens += solve_res.get("completion_tokens", 0)
            total_query_cost += solve_res.get("cost", 0.0)
            total_query_api_calls += solve_res.get("api_calls", 0)
            
            is_correct = False
            is_valid = False
            validation_reason = None
            val_prompt_tokens = 0
            val_completion_tokens = 0
            validation_api_calls = 0
            val_cost = 0.0
            
            if gt_answer is not None:
                domain_total += 1
                is_correct = compare_answers(final_answer, gt_answer)
                if is_correct:
                    domain_correct += 1
                    is_valid = True
                else:
                    if final_answer is not None:
                        logging.info("   Exact match failed. Querying LLM validator...")
                        start_val_prompt = token_usage.get("prompt_tokens", 0)
                        start_val_comp = token_usage.get("completion_tokens", 0)
                        start_val_api = token_usage.get("api_calls", 0)
                        
                        is_valid, validation_reason = validate_plan_llm(query, final_answer)
                        
                        val_prompt_tokens = token_usage.get("prompt_tokens", 0) - start_val_prompt
                        val_completion_tokens = token_usage.get("completion_tokens", 0) - start_val_comp
                        validation_api_calls = token_usage.get("api_calls", 0) - start_val_api
                        val_cost = calculate_cost(val_prompt_tokens, val_completion_tokens)
                        
                        # Add validation metrics to the query stats
                        total_query_prompt_tokens += val_prompt_tokens
                        total_query_completion_tokens += val_completion_tokens
                        total_query_cost += val_cost
                        total_query_api_calls += validation_api_calls
                        
                        if is_valid:
                            logging.info("   ✅ LLM validator passed! Alternative valid plan found.")
                        else:
                            logging.info(f"   ❌ LLM validator failed: {validation_reason}")
                            
                if is_valid:
                    domain_valid += 1
                    
            results.append({
                "example_id": example_id,
                "prompt_0shot": query,
                "predicted_initial_state": solve_res["predicted_initial_state"],
                "predicted_goal_state": solve_res["predicted_goal_state"],
                "plan": solve_res["plan"],
                "final_answer": final_answer,
                "ground_truth_answer": gt_answer,
                "failure_reason": solve_res["failure_reason"],
                "time_taken": time_taken,
                "correct": is_correct,
                "valid": is_valid,
                "validation_reason": validation_reason,
                "prompt_tokens": solve_res.get("prompt_tokens", 0) + val_prompt_tokens,
                "completion_tokens": solve_res.get("completion_tokens", 0) + val_completion_tokens,
                "api_calls": solve_res.get("api_calls", 0) + validation_api_calls,
                "cost": solve_res.get("cost", 0.0) + val_cost
            })
            
        with open(domain_output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"💾 Test results saved to {domain_output_path}")
        
        if domain_total > 0:
            em_acc = 100.0 * domain_correct / domain_total
            sr_acc = 100.0 * domain_valid / domain_total
            domain_total_prompt = build_prompt_tokens + total_query_prompt_tokens
            domain_total_completion = build_completion_tokens + total_query_completion_tokens
            domain_total_cost = build_cost + total_query_cost
            domain_total_api_calls = build_api_calls + total_query_api_calls
            
            logging.info(f"✅ Domain '{domain}' evaluation finished:")
            logging.info(f"   Exact Match Accuracy: {em_acc:.2f}% ({domain_correct}/{domain_total})")
            logging.info(f"   Success Rate (Validity): {sr_acc:.2f}% ({domain_valid}/{domain_total})")
            logging.info(f"💰 Domain Cost: Build = ${build_cost:.4f}, Query = ${total_query_cost:.4f}, Total = ${domain_total_cost:.4f}")
            logging.info(f"📊 Domain Tokens: Build = P:{build_prompt_tokens}/C:{build_completion_tokens}, Query = P:{total_query_prompt_tokens}/C:{total_query_completion_tokens}")
            logging.info(f"📞 Domain API Calls: Build = {build_api_calls}, Query = {total_query_api_calls}, Total = {domain_total_api_calls}")
            
            summary_results[domain] = {
                "total": domain_total,
                "exact_match_correct": domain_correct,
                "exact_match_accuracy": f"{em_acc:.2f}%",
                "success_rate_correct": domain_valid,
                "success_rate": f"{sr_acc:.2f}%",
                "build_prompt_tokens": build_prompt_tokens,
                "build_completion_tokens": build_completion_tokens,
                "build_api_calls": build_api_calls,
                "build_cost_usd": build_cost,
                "query_prompt_tokens": total_query_prompt_tokens,
                "query_completion_tokens": total_query_completion_tokens,
                "query_api_calls": total_query_api_calls,
                "query_cost_usd": total_query_cost,
                "total_prompt_tokens": domain_total_prompt,
                "total_completion_tokens": domain_total_completion,
                "total_api_calls": domain_total_api_calls,
                "total_cost_usd": domain_total_cost
            }

            # ✅ Save summary after EVERY domain so progress is never lost
            summary_path = os.path.join(args.output_dir, "accuracy_summary.json")
            saved = save_summary(summary_path, summary_results)
            n_saved = len(saved["domains"])
            logging.info(f"💾 Summary updated ({n_saved} domain(s) stored) → {summary_path}")

    # Final summary log
    summary_path = os.path.join(args.output_dir, "accuracy_summary.json")
    final = save_summary(summary_path, summary_results)  # ensure final write
    meta = final["metadata"]
    logging.info(f"💾 Final summary saved to {summary_path}")
    logging.info(f"🎯 Overall exact match accuracy: {meta['overall_exact_match_accuracy']} ({meta['overall_correct_em']}/{meta['overall_total']})")
    logging.info(f"🎯 Overall success rate: {meta['overall_success_rate']} ({meta['overall_correct_sr']}/{meta['overall_total']})")

if __name__ == "__main__":
    main()
