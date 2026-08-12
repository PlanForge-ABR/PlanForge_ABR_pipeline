import os
import sys
import time
import json
import argparse
import logging

# Add parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from scope.workflow_acp import SCOPESolverBuilder, StandaloneSCOPESolver
except ImportError:
    from workflow_acp import SCOPESolverBuilder, StandaloneSCOPESolver

# Setup logging
os.makedirs("scope/results/acp_bench", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scope/results/acp_bench/run_acp.log"),
        logging.StreamHandler()
    ]
)

def _normalize_yes(value):
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"yes", "y", "true", "t", "1"}

def discover_domains(test_root):
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test root directory not found: {test_root}")
    
    domains = []
    for fname in os.listdir(test_root):
        if fname.endswith("-test.json"):
            domain = fname[:-10]  # Remove -test.json
            if domain:
                domains.append(domain)
    return sorted(list(set(domains)))

def find_exemplar(test_data):
    # Search for the first positive case ("yes") with a non-empty plan list
    for example in test_data:
        ans = example.get("answer")
        plan = example.get("sample_plan")
        if _normalize_yes(ans) and isinstance(plan, list) and len(plan) > 0:
            return example
    # Fallback to any case with a non-empty plan list
    for example in test_data:
        plan = example.get("sample_plan")
        if isinstance(plan, list) and len(plan) > 0:
            return example
    return None

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
    merged_domains.update(summary_results)   # overwrite/add domains we just ran

    overall_total   = sum(d["total"]   for d in merged_domains.values())
    overall_correct = sum(d["correct"] for d in merged_domains.values())
    overall_acc = 100.0 * overall_correct / overall_total if overall_total > 0 else 0.0

    overall_build_prompt = sum(d["build_prompt_tokens"]     for d in merged_domains.values())
    overall_build_comp   = sum(d["build_completion_tokens"] for d in merged_domains.values())
    overall_build_api_calls = sum(d.get("build_api_calls", 0) for d in merged_domains.values())
    overall_build_cost   = sum(d["build_cost_usd"]          for d in merged_domains.values())
    overall_query_prompt = sum(d["query_prompt_tokens"]     for d in merged_domains.values())
    overall_query_comp   = sum(d["query_completion_tokens"] for d in merged_domains.values())
    overall_query_api_calls = sum(d.get("query_api_calls", 0) for d in merged_domains.values())
    overall_query_cost   = sum(d["query_cost_usd"]          for d in merged_domains.values())
    overall_total_prompt = overall_build_prompt + overall_query_prompt
    overall_total_comp   = overall_build_comp   + overall_query_comp
    overall_total_api_calls = overall_build_api_calls + overall_query_api_calls
    overall_total_cost   = overall_build_cost   + overall_query_cost

    summary_output = {
        "metadata": {
            "model": "gpt-5.4",
            "overall_accuracy": f"{overall_acc:.2f}%",
            "overall_total":           overall_total,
            "overall_correct":         overall_correct,
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
    parser = argparse.ArgumentParser(description="Run Standalone SCOPE on test baseline dataset for all domains.")
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain name (e.g. blocksworld, ferry). Use 'all' (default) for all discovered domains."
    )
    parser.add_argument(
        "--test_root",
        default="data/test_baseline",
        help="Root directory containing baseline test datasets."
    )
    parser.add_argument(
        "--output_dir",
        default="scope/results/acp_bench",
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
        
    logging.info(f"🚀 Starting Standalone SCOPE execution on {len(domains)} domains: {', '.join(domains)}")
    
    overall_total = 0
    overall_correct = 0
    summary_results = {}
    
    for domain in domains:
        logging.info(f"\n===== Domain: {domain} =====")
        
        # Load test baseline json data
        test_path = os.path.join(args.test_root, f"{domain}-test.json")
        if not os.path.exists(test_path):
            logging.error(f"Test file not found: {test_path}")
            continue
            
        with open(test_path, "r", encoding="utf-8") as fh:
            test_data = json.load(fh)
            
        # Find exemplar
        exemplar_example = find_exemplar(test_data)
        if not exemplar_example:
            logging.error(f"Could not find any positive exemplar query with a valid plan in dataset {test_path}!")
            continue
            
        q_ex = f"Context:\n{exemplar_example.get('context','')}\n\nQuestion:\n{exemplar_example.get('inputs','')}"
        s_ex = str(exemplar_example.get("sample_plan"))
        
        logging.info(f"Using exemplar ID: {exemplar_example.get('id')} with plan length: {len(exemplar_example.get('sample_plan'))}")
        
        # Build solver using the exemplar
        try:
            builder = SCOPESolverBuilder(domain, q_ex, s_ex)
            config = builder.build()
            solver = StandaloneSCOPESolver(config)
        except Exception as e:
            logging.error(f"Failed to build solver for domain {domain}: {e}")
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
        domain_correct = 0
        total_query_prompt_tokens = 0
        total_query_completion_tokens = 0
        total_query_cost = 0.0
        total_query_api_calls = 0
        
        for idx, example in enumerate(test_data, start=1):
            example_id = example.get("id")
            context = example.get("context", "")
            inputs = example.get("inputs", "")
            gt_answer = example.get("answer")
            
            logging.info(f"🔄 [{idx}/{len(test_data)}] example_id={example_id}")
            
            start_time = time.time()
            solve_res = solver.solve(context, inputs)
            time_taken = time.time() - start_time
            
            final_answer = solve_res["final_answer"]
            
            # Accumulate query stats
            total_query_prompt_tokens += solve_res.get("prompt_tokens", 0)
            total_query_completion_tokens += solve_res.get("completion_tokens", 0)
            total_query_cost += solve_res.get("cost", 0.0)
            total_query_api_calls += solve_res.get("api_calls", 0)
            
            is_correct = False
            if gt_answer is not None:
                domain_total += 1
                is_correct = _normalize_yes(gt_answer) == _normalize_yes(final_answer)
                if is_correct:
                    domain_correct += 1
                    
            results.append({
                "example_id": example_id,
                "context": context,
                "question": inputs,
                "predicted_initial_state": solve_res["predicted_initial_state"],
                "predicted_goal_state": solve_res["predicted_goal_state"],
                "plan": solve_res["plan"],
                "final_answer": final_answer,
                "ground_truth_answer": gt_answer,
                "failure_reason": solve_res["failure_reason"],
                "time_taken": time_taken,
                "correct": is_correct,
                "prompt_tokens": solve_res.get("prompt_tokens", 0),
                "completion_tokens": solve_res.get("completion_tokens", 0),
                "api_calls": solve_res.get("api_calls", 0),
                "cost": solve_res.get("cost", 0.0)
            })
            
        with open(domain_output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"💾 Test results saved to {domain_output_path}")
        
        if domain_total > 0:
            acc = 100.0 * domain_correct / domain_total
            domain_total_prompt = build_prompt_tokens + total_query_prompt_tokens
            domain_total_completion = build_completion_tokens + total_query_completion_tokens
            domain_total_cost = build_cost + total_query_cost
            domain_total_api_calls = build_api_calls + total_query_api_calls
            
            logging.info(f"✅ Accuracy for domain '{domain}': {acc:.2f}% ({domain_correct}/{domain_total})")
            logging.info(f"💰 Domain Cost: Build = ${build_cost:.4f}, Query = ${total_query_cost:.4f}, Total = ${domain_total_cost:.4f}")
            logging.info(f"📊 Domain Tokens: Build = P:{build_prompt_tokens}/C:{build_completion_tokens}, Query = P:{total_query_prompt_tokens}/C:{total_query_completion_tokens}")
            logging.info(f"📞 Domain API Calls: Build = {build_api_calls}, Query = {total_query_api_calls}, Total = {domain_total_api_calls}")
            
            summary_results[domain] = {
                "total": domain_total,
                "correct": domain_correct,
                "accuracy": f"{acc:.2f}%",
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
            overall_total += domain_total
            overall_correct += domain_correct
            
            # Save summary after every domain so progress is never lost
            summary_path = os.path.join(args.output_dir, "accuracy_summary.json")
            save_summary(summary_path, {domain: summary_results[domain]})
            
    # Save final overall summary
    summary_path = os.path.join(args.output_dir, "accuracy_summary.json")
    final = save_summary(summary_path, summary_results)
    meta = final["metadata"]
    logging.info(f"💾 Final summary saved to {summary_path}")
    logging.info(f"🎯 Overall test accuracy: {meta['overall_accuracy']} ({meta['overall_correct']}/{meta['overall_total']})")

if __name__ == "__main__":
    main()
