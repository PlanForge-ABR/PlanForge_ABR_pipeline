#!/usr/bin/env python3
"""
Run NL2State + Search on a single domain using a pre-optimised DSPy module.

You can choose between:
- a domain-specific optimised module (from ``src/<domain>/<domain>_optimized_module.json``), or
- the multi-domain optimised module (``src/multi_domain_optimized_module.json``).

This script:
1. Loads the chosen DSPy module.
2. Runs NL→state on the test set for the given domain.
3. Runs search for each test example.
4. Saves per-example results and aggregated accuracy for that domain.

Examples:
- Domain-optimised module:
  python main.py --domain blocksworld --module_source domain

- Multi-domain optimised module:
  python main.py --domain blocksworld --module_source multi
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from src.search import Search
from src.search import Search
from src import results as results_module
from src.utils import setup_logging
import logging


def _discover_domains(test_root: str) -> List[str]:
	"""Discover domains from filenames in the test data directory."""
	if not os.path.isdir(test_root):
		raise FileNotFoundError(f"Test root directory not found: {test_root}")

	domains: List[str] = []
	for fname in os.listdir(test_root):
		if fname.endswith(".json"):
			# specific check for the user's current naming convention in test_baseline
			if fname.endswith("-test.json"):
				domain = fname[:-10] # remove -test.json
				if domain:
					domains.append(domain)
			elif "-test-dev" in fname:
				domain = fname.split("-test-dev", 1)[0]
				if domain:
					domains.append(domain)
	return sorted(set(domains))


def _load_test_data(domain: str, test_root: str, test_path: str | None = None) -> Tuple[List[Dict[str, Any]], str]:
	"""Load test data for a domain."""
	if test_path:
		path = test_path
	else:
		# Try the new standard first
		p1 = os.path.join(test_root, f"{domain}-test.json")
		# Then the old standard
		p2 = os.path.join(test_root, f"{domain}-test-dev_08_22_2026.json")
		
		if os.path.exists(p1):
			path = p1
		elif os.path.exists(p2):
			path = p2
		else:
			# Fallback or error
			path = p1 # let it fail with this path in error message or check both

	if not os.path.exists(path):
		raise FileNotFoundError(f"Test file not found for domain '{domain}'. Checked: {path}")

	with open(path, "r", encoding="utf-8") as fh:
		data = json.load(fh)

	if not isinstance(data, list):
		# Handle wrapped format commonly found in some datasets
		if isinstance(data, dict) and 'examples' in data:
			data = data['examples']
		elif isinstance(data, dict) and 'data' in data:
			data = data['data']
		else:
			raise ValueError(f"Expected a list in test file {path}")

	return data, path


def _build_nl2state_processor(
	domain: str,
	module_source: str,
	model: str,
	model_name: str,
	src_path: str,
	module_file: str | None,
):
	"""Instantiate the appropriate NL2StateProcessor and load the optimised module."""
	if module_source == "domain":
		# Per-domain DSPy optimisation (src/nl2state.py)
		from src import nl2state as nl_mod

		processor = nl_mod.NL2StateProcessor(
			domain=domain,
			model=model,
			model_name=model_name,
			src_path=src_path,
		)
		filename = module_file or f"{domain}_optimized_module.json"
	else:
		# Multi-domain DSPy optimisation (src/nl2state_2.py)
		from src import nl2state_2 as nl_mod

		processor = nl_mod.NL2StateProcessor(
			domain=domain,
			model=model,
			model_name=model_name,
			src_path=src_path,
		)
		filename = module_file or "multi_domain_optimized_module.json"

	processor.load_optimized_module(filename)
	return processor, filename


def _normalize_yes(value: Any) -> bool:
	if value is None:
		return False
	s = str(value).strip().lower()
	return s in {"yes", "y", "true", "t", "1"}


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Run NL2State + Search on test data for a single domain using a saved DSPy module.",
	)
	parser.add_argument(
		"--domain",
		default="all",
		help="Domain name (e.g., blocksworld, ferry, logistics, ...). "
		     "Use 'all' (default) to run for every domain found in the test set.",
	)
	parser.add_argument(
		"--module_source",
		choices=["domain", "multi"],
		default="domain",
		help="Which optimised DSPy module to use: 'domain' for src/<domain>/<domain>_optimized_module.json, "
		     "'multi' for src/multi_domain_optimized_module.json.",
	)
	parser.add_argument(
		"--module_file",
		help="Optional override for the optimised module filename.",
	)
	parser.add_argument(
		"--test_data",
		help="Optional path to a test dataset JSON (only used when running a single domain).",
	)
	parser.add_argument(
		"--test_root",
		default=os.path.join("data", "test"),
		help="Root directory containing test files (default: data/test).",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=60,
		help="Timeout for search in seconds (default: 600).",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit number of examples to run (default: runs all).",
	)
	parser.add_argument(
		"--strategy",
		choices=["bfs", "astar", "a*", "a-star", "greedy_bfs"],
		default="astar",
		help="Search strategy (default: astar).",
	)
	parser.add_argument(
		"--model",
		default="openai",
		choices=["openai", "gemini", "grok", "rits"],
		help="Model provider used by DSPy when running the optimised module.",
	)
	parser.add_argument(
		"--model_name",
		default="gpt-5.1",
		help="Model name (e.g., gpt-5.1, gemini-2.5-flash).",
	)
	parser.add_argument(
		"--src",
		default="./src",
		help="Path to src directory containing domain folders and modules (default: ./src).",
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		help="Directory to save results (defaults to --src if not specified).",
	)
	parser.add_argument(
		"--domain_output_file",
		default=None,
		help="Filename for per-domain search results JSON.",
	)
	parser.add_argument(
		"--overall_output_file",
		default=None,
		help="Filename for the aggregated accuracy summary JSON.",
	)

	args = parser.parse_args()

	# Determine which domains to run
	if args.domain and args.domain.lower() != "all":
		domains = [args.domain]
	else:
		domains = _discover_domains(args.test_root)
		if not domains:
			raise SystemExit(f"No domains discovered in test root {args.test_root}")

	setup_logging()

	print(f"🚀 Running NL2State + Search")
	logging.info(f"🚀 Running NL2State + Search")
	print(f"📦 Module source: {args.module_source}")
	logging.info(f"📦 Module source: {args.module_source}")
	print(f"🌐 Domains: {', '.join(domains)}")
	logging.info(f"🌐 Domains: {', '.join(domains)}")

	# Results filenames shared across all domains for this run
	mode_tag = "domain" if args.module_source == "domain" else "multi"
	
	# Determine output root and filenames
	output_root = args.output_dir if args.output_dir else args.src
	search_result_filename = args.domain_output_file if args.domain_output_file else f"search_result_test_{mode_tag}.json"
	summary_out = args.overall_output_file if args.overall_output_file else f"accuracy_summary_test_{mode_tag}.json"
	dom_summary_out = f"search_accuracy_test_{mode_tag}.json" # Not exposed for now, derived similar to others

	overall_total = 0
	overall_correct = 0

	for domain in domains:
		print(f"\n===== Domain: {domain} =====")
		logging.info(f"\n===== Domain: {domain} =====")

		# 1. Load test data for this domain
		test_path_override = args.test_data if len(domains) == 1 and args.test_data else None
		test_data, test_path = _load_test_data(domain, args.test_root, test_path_override)
		
		if args.limit and len(test_data) > args.limit:
			test_data = test_data[:args.limit]
			print(f"⚠️ Limiting to {args.limit} examples.")
			logging.info(f"⚠️ Limiting to {args.limit} examples.")
			
		print(f"📥 Loaded {len(test_data)} test examples from {test_path}")
		logging.info(f"📥 Loaded {len(test_data)} test examples from {test_path}")

		# 2. Build NL2State processor and load the optimised module
		processor, module_filename = _build_nl2state_processor(
			domain=domain,
			module_source=args.module_source,
			model=args.model,
			model_name=args.model_name,
			src_path=args.src,
			module_file=args.module_file,
		)

		print(f"✅ Loaded optimised DSPy module for {domain}: {module_filename}")
		logging.info(f"✅ Loaded optimised DSPy module for {domain}: {module_filename}")

		# 3. Run NL2State + Search for each test example
		# 3. Run NL2State + Search for each test example
		domain_dir = os.path.join(output_root, domain)
		os.makedirs(domain_dir, exist_ok=True)
		search_result_path = os.path.join(domain_dir, search_result_filename)

		results: List[Dict[str, Any]] = []
		total = 0
		correct = 0

		for idx, example in enumerate(test_data, start=1):
			example_id = example.get("id")
			context = example.get("context", "")
			question = example.get("question") or example.get("inputs", "")
			gt_answer = example.get("answer")

			print(f"🔄 [{idx}/{len(test_data)}] example_id={example_id}")
			logging.info(f"🔄 [{idx}/{len(test_data)}] example_id={example_id}")

			# NL → state using the optimised DSPy module
			nl_retry_count = 0
			max_nl_retries = 3
			nl2_result = None
			nl2_error = None

			while nl_retry_count < max_nl_retries:
				try:
					nl2_result = processor.process_example(
						context=context,
						inputs=question,
						example_id=str(example_id) if example_id is not None else None,
					)
					break
				except Exception as e:
					nl_retry_count += 1
					nl2_error = str(e)
					print(f"⚠️ NL2State failed (attempt {nl_retry_count}/{max_nl_retries}): {e}")
					logging.warning(f"⚠️ NL2State failed (attempt {nl_retry_count}/{max_nl_retries}): {e}")
					time.sleep(1) # brief pause before retry

			if nl2_result:
				pred_initial = nl2_result["predicted_initial_state"]
				pred_goal = nl2_result["predicted_goal_state"]

				# Search in the planning domain
				searcher = Search(domain, pred_initial, pred_goal, timeout=args.timeout)
				try:
					found, plan, info = searcher.search(strategy=args.strategy, return_plan=True)
					final_answer = "yes" if found else "no"
					failure_reason = info.get("reason") if not found else None
				except Exception as e:
					print(f"❌ Search failed with exception: {e}")
					logging.error(f"❌ Search failed with exception: {e}")
					found = False
					plan = []
					final_answer = "no"
					failure_reason = f"search exception: {str(e)}"
			else:
				# NL2State failed after retries
				print(f"❌ NL2State failed completely for example {example_id}")
				logging.error(f"❌ NL2State failed completely for example {example_id}")
				pred_initial = None
				pred_goal = None
				found = False
				plan = []
				final_answer = "wrongly answered" # As requested by user
				failure_reason = f"nl2state failed: {nl2_error}"

			# Track simple accuracy against ground-truth yes/no if available
			if gt_answer is not None:
				total += 1
				if _normalize_yes(gt_answer) == _normalize_yes(final_answer):
					correct += 1

			results.append(
				{
					"example_id": nl2_result.get("example_id", example_id) if nl2_result else example_id,
					"context": context,
					"question": question,
					"predicted_initial_state": pred_initial,
					"predicted_goal_state": pred_goal,
					"plan": plan,
					"final_answer": final_answer,
					"ground_truth_answer": gt_answer,
					"failure_reason": failure_reason,
				}
			)

		with open(search_result_path, "w", encoding="utf-8") as fh:
			json.dump(results, fh, indent=2)
		print(f"💾 Test search results saved to {search_result_path}")
		logging.info(f"💾 Test search results saved to {search_result_path}")

		if total > 0:
			acc = 100.0 * correct / total
			print(f"✅ Local test accuracy for domain '{domain}': {acc:.2f}% ({correct}/{total})")
			logging.info(f"✅ Local test accuracy for domain '{domain}': {acc:.2f}% ({correct}/{total})")

		overall_total += total
		overall_correct += correct

	# 4. Compute and save aggregated accuracy across all domains using src/results.py
	print("📊 Computing per-domain and overall accuracy (split=test)...")
	logging.info("📊 Computing per-domain and overall accuracy (split=test)...")
	results_module.main(
		split="test",
		omit_per_example=False,
		out=search_result_filename,
		summary_out=summary_out,
		dom_summary_out=dom_summary_out,
		data_root=args.test_root,
		results_dir=output_root,
	)

	if overall_total > 0:
		acc = 100.0 * overall_correct / overall_total
		print(f"\n🎯 Overall test accuracy across domains: {acc:.2f}% ({overall_correct}/{overall_total})")
		logging.info(f"\n🎯 Overall test accuracy across domains: {acc:.2f}% ({overall_correct}/{overall_total})")

	print("\n🎉 Done.")
	logging.info("\n🎉 Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
