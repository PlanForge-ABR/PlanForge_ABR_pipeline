#!/usr/bin/env python3
"""Evaluate PlanForge NaturalPlan outputs with the SCOPE success-rate protocol.

The protocol is intentionally identical to ``scope/run_scope_natural_plan.py``:
an exact (whitespace-normalized) match to the golden plan is accepted directly;
every other non-empty prediction is assessed by ``validate_plan_llm``.  The
validator is the GPT-5.4 LLM-as-judge used by the SCOPE NaturalPlan runner.

By default this evaluates the first 150 numerically ordered examples in each
domain, matching the 450-instance NaturalPlan comparison.  Results are saved
in the PlanForge results directory, without modifying the source result files.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

DEFAULT_RESULTS_DIR = Path("natural_plan_results_planforge")
DEFAULT_DATA_DIR = Path("data/natural-plan")
DEFAULT_DOMAINS = ("calendar_scheduling", "meeting_planning", "trip_planning")


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """The GPT-5.4 pricing function used by the SCOPE NaturalPlan runner."""
    return (prompt_tokens * 2.50 + completion_tokens * 15.00) / 1_000_000.0


def compare_answers(pred: Any, gt: Any) -> bool:
    """Copy of SCOPE's whitespace-normalized exact-match comparison."""
    if pred is None or gt is None:
        return False
    if isinstance(pred, list) and isinstance(gt, list):
        return [str(x).strip().lower() for x in pred] == [str(x).strip().lower() for x in gt]
    pred_str = "\n".join(str(x).strip().lower() for x in pred) if isinstance(pred, list) else str(pred).strip().lower()
    gt_str = "\n".join(str(x).strip().lower() for x in gt) if isinstance(gt, list) else str(gt).strip().lower()
    return re.sub(r"\s+", " ", pred_str).strip() == re.sub(r"\s+", " ", gt_str).strip()


def get_scope_llm_judge() -> tuple[Any, dict[str, int]]:
    """Lazily load SCOPE's judge so dry runs need no OpenAI dependency."""
    try:
        from scope.workflow_natural_plan import token_usage, validate_plan_llm
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The LLM judge requires the project's OpenAI dependency. "
            "Install the project requirements, then rerun without --skip-llm-judge."
        ) from error
    return validate_plan_llm, token_usage


def numeric_key(value: str) -> tuple[int, str]:
    """Order NaturalPlan IDs such as ``meeting_planning_example_12`` safely."""
    suffix = value.rsplit("_", 1)[-1]
    return (int(suffix), value) if suffix.isdigit() else (10**18, value)


def plan_to_text(plan: Any) -> Optional[str]:
    """Match the representation used by SCOPE's exact-match helper."""
    if plan is None:
        return None
    if isinstance(plan, list):
        return "\n".join(str(step) for step in plan)
    return str(plan)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_domain(
    domain: str,
    results_dir: Path,
    data_dir: Path,
    limit: Optional[int],
    skip_llm_judge: bool,
) -> dict[str, Any]:
    source_path = results_dir / f"{domain}_eval_results.json"
    dataset_path = data_dir / f"{domain}.json"
    source = load_json(source_path)
    dataset = load_json(dataset_path)
    predictions = source.get("predictions", {})
    if not isinstance(predictions, dict) or not isinstance(dataset, dict):
        raise ValueError(f"Expected prediction and dataset dictionaries for {domain}.")

    example_ids = sorted(predictions, key=numeric_key)
    if limit is not None:
        example_ids = example_ids[:limit]
    print(f"\n[{domain}] Evaluating {len(example_ids)} PlanForge plans...")

    entries: list[dict[str, Any]] = []
    exact_match_correct = 0
    success_rate_correct = 0
    judged_count = 0
    judge_prompt_tokens = 0
    judge_completion_tokens = 0
    judge_api_calls = 0
    validate_plan_llm = None
    token_usage: Optional[dict[str, int]] = None
    if not skip_llm_judge:
        validate_plan_llm, token_usage = get_scope_llm_judge()

    for index, example_id in enumerate(example_ids, start=1):
        result = predictions[example_id]
        example = dataset.get(example_id)
        if example is None:
            raise KeyError(f"{example_id} is absent from {dataset_path}")

        prediction = plan_to_text(result.get("plan"))
        golden_plan = example.get("golden_plan")
        exact_match = compare_answers(prediction, golden_plan)
        valid = exact_match
        validation_reason: Optional[str] = "Exact match to golden plan." if exact_match else None
        validation_method = "exact_match" if exact_match else "not_evaluated"
        validation_error: Optional[str] = None
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0

        if exact_match:
            exact_match_correct += 1
        elif prediction is None or not prediction.strip():
            validation_method = "empty_prediction"
            validation_reason = "No PlanForge plan was returned."
        elif skip_llm_judge:
            validation_method = "llm_judge_skipped"
            validation_reason = "Non-exact plan was not judged (--skip-llm-judge)."
        else:
            validation_method = "llm_judge"
            judged_count += 1
            print(f"[{domain}] [{index}/{len(example_ids)}] Exact match failed; running LLM judge for {example_id}...")
            assert validate_plan_llm is not None and token_usage is not None
            start_prompt = token_usage.get("prompt_tokens", 0)
            start_completion = token_usage.get("completion_tokens", 0)
            start_calls = token_usage.get("api_calls", 0)
            try:
                valid, validation_reason = validate_plan_llm(example.get("prompt_0shot", ""), prediction)
            except Exception as error:  # Defensive: preserve partial evaluations.
                valid = False
                validation_error = str(error)
                validation_reason = f"LLM judge failed: {error}"
            prompt_tokens = token_usage.get("prompt_tokens", 0) - start_prompt
            completion_tokens = token_usage.get("completion_tokens", 0) - start_completion
            api_calls = token_usage.get("api_calls", 0) - start_calls
            judge_prompt_tokens += prompt_tokens
            judge_completion_tokens += completion_tokens
            judge_api_calls += api_calls
            # The shared SCOPE helper converts judge transport failures to
            # ``(False, '<error>')``.  Do not silently report those as failed
            # plans: no SR is meaningful until every non-exact plan is judged.
            if validation_reason and validation_reason.lower() in {"connection error", "api connection error"}:
                raise RuntimeError(
                    f"LLM judge did not evaluate {example_id}: {validation_reason}. "
                    "No results file was written."
                )

        if valid:
            success_rate_correct += 1
        entries.append(
            {
                "example_id": example_id,
                "prediction": result.get("plan"),
                "golden_plan": golden_plan,
                "exact_match": exact_match,
                "valid": valid,
                "validation_method": validation_method,
                "validation_reason": validation_reason,
                "validation_error": validation_error,
                "judge_prompt_tokens": prompt_tokens,
                "judge_completion_tokens": completion_tokens,
                "judge_api_calls": api_calls,
                "judge_cost_usd": calculate_cost(prompt_tokens, completion_tokens),
            }
        )
        if index % 25 == 0 or index == len(example_ids):
            print(
                f"[{domain}] Progress: {index}/{len(example_ids)} | "
                f"exact={exact_match_correct} | valid={success_rate_correct} | judged={judged_count}"
            )

    total = len(entries)
    summary = {
        "source_results_file": str(source_path),
        "dataset_file": str(dataset_path),
        "total": total,
        "exact_match_correct": exact_match_correct,
        "exact_match_accuracy": f"{100 * exact_match_correct / total:.2f}%" if total else "0.00%",
        "success_rate_correct": success_rate_correct,
        "success_rate": f"{100 * success_rate_correct / total:.2f}%" if total else "0.00%",
        "llm_judged_non_exact": judged_count,
        "judge_prompt_tokens": judge_prompt_tokens,
        "judge_completion_tokens": judge_completion_tokens,
        "judge_api_calls": judge_api_calls,
        "judge_cost_usd": calculate_cost(judge_prompt_tokens, judge_completion_tokens),
        "results": entries,
    }
    print(
        f"[{domain}] Complete: exact match {summary['exact_match_accuracy']} "
        f"({exact_match_correct}/{total}), SR {summary['success_rate']} "
        f"({success_rate_correct}/{total})."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score PlanForge NaturalPlan outputs with SCOPE's SR protocol.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--domain", default="all", help="Comma-separated domain names, or 'all'.")
    parser.add_argument("--limit", type=int, default=150, help="Examples per domain; use 0 for all examples.")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <results-dir>/scope_sr_evaluation.json.")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Dry run: do not judge non-exact plans.")
    args = parser.parse_args()

    domains = DEFAULT_DOMAINS if args.domain.lower() == "all" else tuple(d.strip() for d in args.domain.split(",") if d.strip())
    limit = None if args.limit == 0 else args.limit
    if limit is not None and limit < 1:
        parser.error("--limit must be positive, or 0 to evaluate every example.")

    print(f"Evaluating {len(domains)} domain(s); limit per domain: {limit if limit is not None else 'all'}.")
    domain_results = {
        domain: evaluate_domain(domain, args.results_dir, args.data_dir, limit, args.skip_llm_judge)
        for domain in domains
    }
    overall_total = sum(domain["total"] for domain in domain_results.values())
    overall_exact = sum(domain["exact_match_correct"] for domain in domain_results.values())
    overall_valid = sum(domain["success_rate_correct"] for domain in domain_results.values())
    output = args.output or args.results_dir / "scope_sr_evaluation.json"
    payload = {
        "metadata": {
            "metric": "SCOPE NaturalPlan success rate (exact match + GPT-5.4 LLM-as-judge)",
            "llm_judge_skipped": args.skip_llm_judge,
            "examples_per_domain": limit if limit is not None else "all",
            "overall_total": overall_total,
            "overall_exact_match_correct": overall_exact,
            "overall_exact_match_accuracy": f"{100 * overall_exact / overall_total:.2f}%" if overall_total else "0.00%",
            "overall_success_rate_correct": overall_valid,
            "overall_success_rate": f"{100 * overall_valid / overall_total:.2f}%" if overall_total else "0.00%",
        },
        "domains": domain_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Overall SR: {payload['metadata']['overall_success_rate']} ({overall_valid}/{overall_total})")
    print(f"Saved SCOPE-compatible PlanForge SR evaluation to {output}")


if __name__ == "__main__":
    main()
