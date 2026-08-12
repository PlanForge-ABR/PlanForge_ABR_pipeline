import os
import sys
import time
import json
import argparse
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from scope.workflow_structured_sat import SCOPESolverBuilder, StandaloneSCOPESolver, calculate_cost, check_sat_solution_satisfies_clauses
except ImportError:
    from workflow_structured_sat import SCOPESolverBuilder, StandaloneSCOPESolver, calculate_cost, check_sat_solution_satisfies_clauses

# Setup logging
os.makedirs("scope/results/structured_sat", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scope/results/structured_sat/run_structured_sat.log"),
        logging.StreamHandler()
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

    overall_total   = sum(d["total"]                for d in merged_domains.values())
    overall_correct = sum(d["exact_match_correct"]  for d in merged_domains.values())
    overall_valid   = sum(d["success_rate_correct"] for d in merged_domains.values())

    overall_em_acc = 100.0 * overall_correct / overall_total if overall_total > 0 else 0.0
    overall_sr_acc = 100.0 * overall_valid   / overall_total if overall_total > 0 else 0.0

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
    parser = argparse.ArgumentParser(description="Run Standalone SCOPE on StructuredSAT datasets.")
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain name (e.g. 3-sat_balanced, automor_balanced). Use 'all' (default)."
    )
    parser.add_argument(
        "--test_root",
        default="data/StructuredSAT",
        help="Root directory containing StructuredSAT domains."
    )
    parser.add_argument(
        "--output_dir",
        default="scope/results/structured_sat",
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
        domains = sorted(list(DOMAIN_SUBDIRS.keys()))
        
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
        
    logging.info(f"🚀 Starting Standalone SCOPE execution on {len(domains)} StructuredSAT domains: {', '.join(domains)}")
    
    overall_total = 0
    overall_correct = 0  # EM: correct SAT/UNSAT prediction
    overall_valid = 0    # SR: correct prediction and valid satisfying assignment
    summary_results = {}
    
    # We can compile the generic solver once and reuse it across all SAT domains to minimize LLM build cost,
    # but we will save the generated solver codes under each domain name in the output directory.
    cached_config = None
    
    for domain in domains:
        logging.info(f"\n===== Domain: {domain} =====")
        
        sub_path = DOMAIN_SUBDIRS.get(domain)
        if not sub_path:
            logging.error(f"Unknown domain mapping for {domain}")
            continue
            
        domain_dir = os.path.join(args.test_root, domain, sub_path)
        if not os.path.exists(domain_dir):
            logging.error(f"Domain directory not found: {domain_dir}")
            continue
            
        # Get CNF files
        cnf_files = sorted([f for f in os.listdir(domain_dir) if f.endswith(".cnf")])
        if not cnf_files:
            logging.error(f"No CNF files found in {domain_dir}")
            continue
            
        # Ground truth is parsed from the path
        if "unsat" in sub_path:
            gt_answer = "unsat"
        else:
            gt_answer = "sat"
            
        logging.info(f"Domain '{domain}' has {len(cnf_files)} instances. Ground Truth: {gt_answer}")
        
        # Build solver (build once and cache, or reuse)
        if cached_config is None:
            # We use a simple 3-variable satisfiable exemplar to construct the solver
            q_ex = json.dumps({
                "variables": 3,
                "clauses": [[1, 2], [-1, 3], [-2, -3]]
            })
            s_ex = [1, -2, 3]
            try:
                builder = SCOPESolverBuilder(domain, q_ex, s_ex)
                cached_config = builder.build()
            except Exception as e:
                import traceback
                logging.error(f"Failed to build generic SAT solver: {e}\n{traceback.format_exc()}")
                continue
                
        # Copy cached config and update domain
        config = cached_config.copy()
        
        build_prompt_tokens = config.get("build_prompt_tokens", 0)
        build_completion_tokens = config.get("build_completion_tokens", 0)
        build_api_calls = config.get("build_api_calls", 0)
        build_cost = config.get("build_cost", 0.0)
        
        # Save solver code checkpoints for this specific domain
        solver_code_dir = os.path.join(args.output_dir, "solver_codes")
        os.makedirs(solver_code_dir, exist_ok=True)
        with open(os.path.join(solver_code_dir, f"{domain}_combinations.py"), "w") as f:
            f.write(config["combinations_func_code"])
        with open(os.path.join(solver_code_dir, f"{domain}_plan.py"), "w") as f:
            f.write(config["plan_func_code"])
        with open(os.path.join(solver_code_dir, f"{domain}_deliver.py"), "w") as f:
            f.write(config["deliver_func_code"])
            
        solver = StandaloneSCOPESolver(config)
        
        if args.limit and len(cnf_files) > args.limit:
            cnf_files = cnf_files[:args.limit]
            logging.info(f"⚠️ Limiting execution to {args.limit} examples.")
            
        domain_output_dir = os.path.join(args.output_dir, "output_full_gpt-5.4")
        os.makedirs(domain_output_dir, exist_ok=True)
        domain_output_path = os.path.join(domain_output_dir, f"{domain}_scope.json")
        
        results = []
        domain_total = 0
        domain_correct = 0  # EM correct SAT/UNSAT
        domain_valid = 0    # SR correct prediction and satisfying assignment
        total_query_prompt_tokens = 0
        total_query_completion_tokens = 0
        total_query_cost = 0.0
        total_query_api_calls = 0
        
        for idx, filename in enumerate(cnf_files, start=1):
            example_id = filename.replace(".cnf", "")
            file_path = os.path.join(domain_dir, filename)
            
            # Parse the CNF file to structured JSON query
            cnf_data = parse_cnf_file(file_path)
            query = json.dumps(cnf_data)
            
            logging.info(f"🔄 [{idx}/{len(cnf_files)}] example_id={example_id} (vars={cnf_data['variables']}, clauses={len(cnf_data['clauses'])})")
            
            start_time = time.time()
            solve_res = solver.solve(query)
            time_taken = time.time() - start_time
            
            final_answer = solve_res["final_answer"]
            plan = solve_res["plan"]
            
            # Accumulate query stats
            total_query_prompt_tokens += solve_res.get("prompt_tokens", 0)
            total_query_completion_tokens += solve_res.get("completion_tokens", 0)
            total_query_cost += solve_res.get("cost", 0.0)
            total_query_api_calls += solve_res.get("api_calls", 0)
            
            is_correct = False
            is_valid = False
            validation_reason = None
            
            # 1. Base comparison: check sat vs unsat
            if final_answer == gt_answer:
                is_correct = True
                is_valid = True
                
            # 2. Math Validator check: if sat is predicted, assignment must satisfy clauses
            if final_answer == "sat":
                if plan is not None:
                    assignment_valid = check_sat_solution_satisfies_clauses(plan, cnf_data["clauses"])
                    if not assignment_valid:
                        logging.warning(f"   ❌ Math Validator failed! Solver returned 'sat' but the plan doesn't satisfy clauses.")
                        is_valid = False
                        validation_reason = "Assignment does not satisfy the CNF clauses."
                    else:
                        logging.info("   ✅ Math Validator passed! assignment satisfies all clauses.")
                else:
                    is_valid = False
                    validation_reason = "Predicted 'sat' but no assignment plan was returned."
            
            if is_correct:
                domain_correct += 1
            if is_valid:
                domain_valid += 1
                
            domain_total += 1
            
            results.append({
                "example_id": example_id,
                "file_path": file_path,
                "parsed_query": cnf_data,
                "predicted_initial_state": solve_res["predicted_initial_state"],
                "predicted_goal_state": solve_res["predicted_goal_state"],
                "plan": plan,
                "final_answer": final_answer,
                "ground_truth_answer": gt_answer,
                "failure_reason": solve_res["failure_reason"],
                "validation_reason": validation_reason,
                "time_taken": time_taken,
                "correct": is_correct,
                "valid": is_valid,
                "prompt_tokens": solve_res.get("prompt_tokens", 0),
                "completion_tokens": solve_res.get("completion_tokens", 0),
                "api_calls": solve_res.get("api_calls", 0),
                "cost": solve_res.get("cost", 0.0)
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
