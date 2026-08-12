import os
import re
import json
import glob

def parse_log_for_api_calls(log_path):
    print(f"Parsing log file: {log_path}")
    if not os.path.exists(log_path):
        print(f"  Warning: Log file not found: {log_path}")
        return {}, {}

    build_counts = {}
    query_counts = {}

    current_domain = None
    current_phase = None
    current_example_id = None

    # Regex patterns
    # Matches: Starting Standalone SCOPE Solver construction for domain: blocksworld...
    # Matches: Starting Standalone SCOPE Solver construction for Natural Plan domain: calendar_scheduling...
    # Matches: Starting Standalone SCOPE Solver construction for StructuredSAT: 3-sat_balanced...
    build_pattern = re.compile(
        r"Starting Standalone SCOPE Solver construction for (?:domain|Natural Plan domain|StructuredSAT):\s*([a-zA-Z0-9_\-]+)"
    )

    # Matches: 🔄 [idx/total] example_id=example_id
    query_pattern = re.compile(r"🔄 \[\d+/\d+\] example_id=([a-zA-Z0-9_\-]+)")

    # Matches end of domain evaluation:
    # Matches: ✅ Accuracy for domain 'blocksworld'
    # Matches: ✅ Domain '3-sat_balanced' evaluation finished:
    end_pattern = re.compile(r"✅\s*(?:Accuracy for domain|Domain)\s*'([a-zA-Z0-9_\-]+)'")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 1. Check build start
            build_match = build_pattern.search(line)
            if build_match:
                current_domain = build_match.group(1)
                current_phase = "build"
                build_counts[current_domain] = 0
                continue

            # 2. Check query start
            query_match = query_pattern.search(line)
            if query_match:
                current_example_id = query_match.group(1)
                current_phase = "query"
                query_counts[(current_domain, current_example_id)] = 0
                continue

            # 3. Check end of domain
            end_match = end_pattern.search(line)
            if end_match:
                current_phase = None
                continue

            # 4. Check for API call
            if "HTTP Request: POST" in line:
                if current_domain is not None:
                    if current_phase == "build":
                        build_counts[current_domain] = build_counts.get(current_domain, 0) + 1
                    elif current_phase == "query" and current_example_id is not None:
                        key = (current_domain, current_example_id)
                        query_counts[key] = query_counts.get(key, 0) + 1

    return build_counts, query_counts

def backfill_benchmark(benchmark_dir, log_filename):
    print(f"\n==================================================")
    print(f"Backfilling benchmark directory: {benchmark_dir}")
    print(f"==================================================")
    
    log_path = os.path.join(benchmark_dir, log_filename)
    build_counts, query_counts = parse_log_for_api_calls(log_path)
    
    print(f"Found build API calls counts: {build_counts}")
    print(f"Found query API calls counts (total tracked: {len(query_counts)})")

    # 1. Update detailed output JSON files
    detail_dir = os.path.join(benchmark_dir, "output_full_gpt-5.4")
    detail_pattern = os.path.join(detail_dir, "*_scope.json")
    detail_files = glob.glob(detail_pattern)

    if not detail_files:
        print(f"  Warning: No detail JSON files found in {detail_dir}")

    for filepath in detail_files:
        filename = os.path.basename(filepath)
        domain = filename[:-11] # strip '_scope.json'
        
        print(f"Processing detail JSON: {filename} (domain: {domain})")
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not isinstance(results, list):
            print(f"  Skipping {filename} (not a list)")
            continue

        for item in results:
            if not isinstance(item, dict):
                continue
            example_id = item.get("example_id")
            
            # Retrieve query calls from parsed logs, default to 1 if not in log but prompt_tokens > 0, else 0
            api_calls = query_counts.get((domain, example_id))
            if api_calls is None:
                p_tokens = item.get("prompt_tokens", 0)
                api_calls = 1 if p_tokens > 0 else 0
            
            item["api_calls"] = api_calls

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved detail updates to: {filepath}")

    # 2. Update accuracy_summary.json
    summary_path = os.path.join(benchmark_dir, "accuracy_summary.json")
    if os.path.exists(summary_path):
        print(f"Processing accuracy summary: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        domains_dict = summary.get("domains", {})
        for domain_name, domain_data in list(domains_dict.items()):
            # A. Get build API calls
            build_api = build_counts.get(domain_name, 0)
            
            # B. Get query API calls from the detail JSON file we just updated
            query_api = 0
            detail_file_path = os.path.join(detail_dir, f"{domain_name}_scope.json")
            if os.path.exists(detail_file_path):
                with open(detail_file_path, "r", encoding="utf-8") as f:
                    detail_results = json.load(f)
                if isinstance(detail_results, list):
                    query_api = sum(item.get("api_calls", 0) for item in detail_results if isinstance(item, dict))

            domain_data["build_api_calls"] = build_api
            domain_data["query_api_calls"] = query_api
            domain_data["total_api_calls"] = build_api + query_api

        # C. Update overall metadata totals
        metadata = summary.get("metadata", {})
        
        overall_build_api = sum(d.get("build_api_calls", 0) for d in domains_dict.values())
        overall_query_api = sum(d.get("query_api_calls", 0) for d in domains_dict.values())
        overall_total_api = overall_build_api + overall_query_api

        metadata["build_api_calls"] = overall_build_api
        metadata["query_api_calls"] = overall_query_api
        metadata["total_api_calls"] = overall_total_api

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary updates to: {summary_path}")
    else:
        print(f"  Warning: accuracy_summary.json not found in {benchmark_dir}")

def main():
    base_dir = "/Users/sarvesh/Desktop/IBM/IBM/scope/results"
    
    # ACP Bench
    backfill_benchmark(os.path.join(base_dir, "acp_bench"), "run_acp.log")
    
    # Natural Plan
    backfill_benchmark(os.path.join(base_dir, "natural_plan"), "run_natural_plan.log")
    
    # Structured SAT
    backfill_benchmark(os.path.join(base_dir, "structured_sat"), "run_structured_sat.log")

if __name__ == "__main__":
    main()
