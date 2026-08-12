"""Compute search accuracy per-domain and overall.

This script looks for domain folders under the `src` package (the directory
containing this file). For each domain that contains `search_result.json` it
computes simple search metrics and writes a `search_accuracy.json` into the
domain folder. It also writes an aggregated `accuracy_summary.json` into the
`src` folder.

Metrics produced per-domain:
- num_examples: total examples processed
- num_solved_by_final_answer: count where `final_answer` normalizes to a
	positive/yes value
- num_with_plan: count where the returned `plan` is non-empty
- final_answer_ratio and plan_ratio: the above counts divided by num_examples

If a domain has no `search_result.json` it will be skipped with a printed
warning.

usage: python results.py [--split train|test|none]
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List
from src.utils import setup_logging


def _normalize_yes(value: Any) -> bool:
	"""Return True if value represents a positive/yes answer.

	Accepts a variety of formats: booleans, numeric flags, and strings like
	'yes'/'no', 'true'/'false'.
	"""
	if value is None:
		return False
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	s = str(value).strip().lower()
	return s in {"yes", "y", "true", "t", "1"}


def _normalize_id(value: Any) -> str | None:
	"""Return a consistent string id for lookup."""
	if value is None:
		return None
	return str(value)


def analyze_domain(domain_dir: str) -> Dict[str, Any]:
	"""Analyze a single domain folder (path -> absolute) and return stats.

	Expects a `search_result.json` file in the directory. Returns a dict with
	computed metrics and a small per-example summary.
	"""
	results_path = os.path.join(domain_dir, "search_result.json")
	if not os.path.exists(results_path):
		raise FileNotFoundError(results_path)

	with open(results_path, "r", encoding="utf-8") as fh:
		data = json.load(fh)

	if not isinstance(data, list):
		raise ValueError(f"expected a list in {results_path}")

	num_examples = 0
	num_yes = 0
	num_with_plan = 0
	per_example: List[Dict[str, Any]] = []
	# ground-truth fields filled by caller into entry.get("_ground_truth_answer")
	num_with_ground_truth = 0
	num_correct_final_answer = 0

	for entry in data:
		num_examples += 1
		example_id = entry.get("example_id")
		final_answer = entry.get("final_answer")
		plan = entry.get("plan", [])

		# plan may be a JSON string in some dumps; try to coerce
		if isinstance(plan, str):
			try:
				plan = json.loads(plan)
			except Exception:
				# leave as string and treat as non-empty if non-empty string
				pass

		has_plan = bool(plan) and (not (isinstance(plan, list) and len(plan) == 0))
		is_yes = _normalize_yes(final_answer)

		if is_yes:
			num_yes += 1
		if has_plan:
			num_with_plan += 1

		# attempt to get ground-truth (may be injected by the caller)
		ground_truth = entry.get("_ground_truth_answer")
		is_correct = None
		if ground_truth is not None:
			num_with_ground_truth += 1
			gt_yes = _normalize_yes(ground_truth)
			is_correct = (gt_yes == is_yes)
			if is_correct:
				num_correct_final_answer += 1

		per_example.append({
			"example_id": example_id,
			"final_answer": final_answer,
			"is_yes": is_yes,
			"has_plan": bool(has_plan),
			"plan_length": len(plan) if isinstance(plan, list) else None,
			"ground_truth": ground_truth,
			"is_correct_final_answer": is_correct,
		})

	stats = {
		"num_examples": num_examples,
		"num_solved_by_final_answer": num_yes,
		"num_with_plan": num_with_plan,
		"final_answer_ratio": (num_yes / num_examples) if num_examples else 0.0,
		"plan_ratio": (num_with_plan / num_examples) if num_examples else 0.0,
		"num_with_ground_truth": num_with_ground_truth,
		"num_correct_final_answer": num_correct_final_answer,
		"final_answer_accuracy_against_ground_truth": (
			(num_correct_final_answer / num_with_ground_truth) if num_with_ground_truth else None
		),
		"per_example": per_example,
	}

	return stats


def _load_ground_truth_map(split: str, domain_name: str, repo_root: str, data_root: str | None = None) -> Dict[str, Any]:
	"""Load a map of example_id -> answer for the given split and domain.

	split: one of 'train', 'test', or 'none'. repo_root is the repository root
	(one level above src dir).
	"""
	if split not in {"train", "test"}:
		return {}

	if data_root:
		train_dir = data_root
	else:
		train_dir = os.path.join(repo_root, "data", split)
	gt_map: Dict[str, Any] = {}
	if not os.path.exists(train_dir):
		return {}

	for fname in os.listdir(train_dir):
		if domain_name in fname and ("training" in fname or "test" in fname or split in fname):
			try:
				with open(os.path.join(train_dir, fname), "r", encoding="utf-8") as f:
					arr = json.load(f)
					for item in arr:
						if isinstance(item, dict) and "id" in item:
							eid = _normalize_id(item.get("id"))
							if eid is not None:
								gt_map[eid] = item.get("answer")
			except Exception:
				# ignore problematic files
				pass

	return gt_map


def main(split: str = "train", omit_per_example: bool = False, out: str = "search_result.json", summary_out: str = "accuracy_summary.json", dom_summary_out: str = "search_accuracy.json", data_root: str | None = None, results_dir: str | None = None) -> int:
	if results_dir:
		src_dir = results_dir
	else:
		src_dir = os.path.dirname(__file__)
	
	repo_root = os.path.dirname(os.path.dirname(__file__)) # Assume repo root is 2 levels up from this file

	domain_dirs: List[str] = []
	for entry in os.listdir(src_dir):
		full = os.path.join(src_dir, entry)
		if os.path.isdir(full):
			# consider it a domain if it contains the requested results file
			if os.path.exists(os.path.join(full, out)):
				domain_dirs.append(full)

	if not domain_dirs:
		print(f"No domain folders with {out} were found.")
		logging.warning(f"No domain folders with {out} were found.")
		return 1

	summary: Dict[str, Any] = {
		"domains": [],
		"total_examples": 0,
		"total_solved_by_final_answer": 0,
		"total_with_plan": 0,
		"total_with_ground_truth": 0,
		"total_correct_final_answer": 0,
	}

	for dom in sorted(domain_dirs):
		domain_name = os.path.basename(dom)
		try:
			# load ground-truth map for requested split
			gt_map = _load_ground_truth_map(split, domain_name, repo_root, data_root=data_root)

			# load domain results and inject ground-truth answers into entries
			# so analyze_domain can compute correctness per-example
			results_path = os.path.join(dom, out)
			with open(results_path, "r", encoding="utf-8") as fh:
				data = json.load(fh)

			# inject ground truth onto entries if available
			for entry in data:
				eid = _normalize_id(entry.get("example_id"))
				if eid in gt_map:
					entry["_ground_truth_answer"] = gt_map[eid]

			# Analyze the loaded `data` (which now may have _ground_truth_answer
			# injected). This avoids touching the original search_result.json and
			# keeps analysis deterministic.
			num_examples = 0
			num_yes = 0
			num_with_plan = 0
			per_example = []
			num_with_ground_truth = 0
			num_correct_final_answer = 0

			for entry in data:
				num_examples += 1
				example_id = entry.get("example_id")
				final_answer = entry.get("final_answer")
				plan = entry.get("plan", [])
				if isinstance(plan, str):
					try:
						plan = json.loads(plan)
					except Exception:
						pass
				has_plan = bool(plan) and (not (isinstance(plan, list) and len(plan) == 0))
				is_yes = _normalize_yes(final_answer)
				if is_yes:
					num_yes += 1
				if has_plan:
					num_with_plan += 1
				ground_truth = entry.get("_ground_truth_answer")
				is_correct = None
				if ground_truth is not None:
					num_with_ground_truth += 1
					gt_yes = _normalize_yes(ground_truth)
					is_correct = (gt_yes == is_yes)
					if is_correct:
						num_correct_final_answer += 1
				per_example.append({
					"example_id": example_id,
					"final_answer": final_answer,
					"is_yes": is_yes,
					"has_plan": bool(has_plan),
					"plan_length": len(plan) if isinstance(plan, list) else None,
					"ground_truth": ground_truth,
					"is_correct_final_answer": is_correct,
				})

			stats = {
				"num_examples": num_examples,
				"num_solved_by_final_answer": num_yes,
				"num_with_plan": num_with_plan,
				"final_answer_ratio": (num_yes / num_examples) if num_examples else 0.0,
				"plan_ratio": (num_with_plan / num_examples) if num_examples else 0.0,
				"num_with_ground_truth": num_with_ground_truth,
				"num_correct_final_answer": num_correct_final_answer,
				"final_answer_accuracy_against_ground_truth": (
					(num_correct_final_answer / num_with_ground_truth) if num_with_ground_truth else None
				),
				"per_example": (per_example if not omit_per_example else []),
			}
		except Exception as exc:  # keep going even if one domain fails
			print(f"skipping {domain_name}: {exc}")
			logging.warning(f"skipping {domain_name}: {exc}")
			continue

		# write per-domain accuracy JSON (omit per_example if very large?)
		out_path = os.path.join(dom, dom_summary_out)
		with open(out_path, "w", encoding="utf-8") as fh:
			json.dump({"domain": domain_name, **stats}, fh, indent=2)

		
		msg = (
			f"Wrote {out_path}: {stats['num_examples']} examples, "
			f"final_answer={stats['final_answer_ratio']:.3f}, plan={stats['plan_ratio']:.3f}"
		)
		print(msg)
		logging.info(msg)

		summary["domains"].append({
			"domain": domain_name,
			"num_examples": stats["num_examples"],
			"final_answer_ratio": stats["final_answer_ratio"],
			"plan_ratio": stats["plan_ratio"],
			"num_with_ground_truth": stats.get("num_with_ground_truth", 0),
			"num_correct_final_answer": stats.get("num_correct_final_answer", 0),
			"final_answer_accuracy_against_ground_truth": stats.get("final_answer_accuracy_against_ground_truth"),
		})

		# accumulate overall totals per domain
		summary["total_examples"] += stats["num_examples"]
		summary["total_solved_by_final_answer"] += stats["num_solved_by_final_answer"]
		summary["total_with_plan"] += stats["num_with_plan"]
		summary["total_with_ground_truth"] += stats.get("num_with_ground_truth", 0)
		summary["total_correct_final_answer"] += stats.get("num_correct_final_answer", 0)

	# finalize overall ratios
	total = summary["total_examples"]
	summary["overall_final_answer_ratio"] = (
		summary["total_solved_by_final_answer"] / total if total else 0.0
	)
	summary["overall_plan_ratio"] = summary["total_with_plan"] / total if total else 0.0
	# overall accuracy against ground-truth (if any ground-truth examples exist)
	total_gt = summary.get("total_with_ground_truth", 0)
	summary["overall_ground_truth_accuracy"] = (
		summary["total_correct_final_answer"] / total_gt if total_gt else None
	)

	summary_path = os.path.join(src_dir, summary_out)
	with open(summary_path, "w", encoding="utf-8") as fh:
		json.dump(summary, fh, indent=2)

	msg_summary = (
		f"Wrote overall summary to {summary_path}: total_examples={total}, "
		f"overall_final_answer_ratio={summary['overall_final_answer_ratio']:.3f}, "
		f"overall_plan_ratio={summary['overall_plan_ratio']:.3f}, "
		f"overall_ground_truth_accuracy={summary['overall_ground_truth_accuracy']}"
	)
	print(msg_summary)
	logging.info(msg_summary)

	return 0


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Compute per-domain and overall search accuracy.")
	parser.add_argument("--split", choices=["train", "test", "none"], default="train",
						help="Which dataset split to use for ground-truth answers (train/test/none)")
	parser.add_argument("--omit-per-example", action="store_true",
						help="Omit per_example entries from per-domain output JSON to save space")
	parser.add_argument("--out", default="search_result.json", type=str, help="Path to save search results")  # IGNORE
	parser.add_argument("--dom_summary_out", default="search_accuracy.json", type=str, help="Path to save accuracy summary")  # IGNORE
	parser.add_argument("--summary_out", default="accuracy_summary.json", type=str, help="Path to save accuracy summary")  # IGNORE
	parser.add_argument("--data_root", default=None, type=str, help="Root directory for ground truth data")
	parser.add_argument("--results_dir", default=None, type=str, help="Directory to scan for results")
	args = parser.parse_args()

	setup_logging()

	raise SystemExit(main(split=args.split, omit_per_example=args.omit_per_example, out=args.out, summary_out=args.summary_out, dom_summary_out=args.dom_summary_out, data_root=args.data_root, results_dir=args.results_dir))
