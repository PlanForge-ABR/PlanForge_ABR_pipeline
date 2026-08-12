from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from calendar_planner import builder_run, normalize_plan, parse_problem, verify_result


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "NaturalPlan"
    / "data"
    / "calendar_scheduling.json"
)


def evaluate(dataset_path: Path, limit: Optional[int] = None, offset: int = 0) -> Dict[str, object]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = list(data.items())[offset:]
    if limit is not None:
        items = items[:limit]

    successes = 0
    exact_matches = 0
    failures = []
    predictions = {}

    for key, instance in items:
        problem = parse_problem(key, instance)
        result = builder_run(key, instance)
        verified = verify_result(problem, result)
        exact = normalize_plan(result.plan) == normalize_plan(instance.get("golden_plan"))

        successes += int(verified)
        exact_matches += int(exact)
        predictions[key] = {
            "status": result.status,
            "plan": result.plan,
            "success": verified,
            "exact_match": exact,
            "reason": result.reason,
        }
        if not verified or not exact:
            failures.append(
                {
                    "key": key,
                    "status": result.status,
                    "success": verified,
                    "exact_match": exact,
                    "reason": result.reason,
                    "prediction": result.plan,
                    "golden_plan": instance.get("golden_plan"),
                }
            )

    total = len(items)
    return {
        "dataset_path": str(dataset_path),
        "total": total,
        "successes": successes,
        "exact_matches": exact_matches,
        "success_rate": successes / total if total else 0.0,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "failures": failures,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ABR calendar-scheduling solver.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("calendar_scheduling_eval_results.json"))
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args()

    report = evaluate(args.dataset, limit=args.limit, offset=args.offset)
    summary = {key: value for key, value in report.items() if key not in {"failures", "predictions"}}
    summary["num_failures_or_em_misses"] = len(report["failures"])
    print(json.dumps(summary, indent=2))

    if not args.no_output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
