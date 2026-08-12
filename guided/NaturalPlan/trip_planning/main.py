from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from runner.evaluation import EvaluationResult, evaluate_prediction, summarize
from runner.executor import prepare_builder_output
from runner.search_runner import format_plan, run_search


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "NaturalPlan" / "data" / "trip_planning.json"


def load_dataset(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline(
    dataset_path: Path,
    limit: int | None,
    offset: int,
    algorithm: str,
    output_path: Path | None,
    include_golden: bool,
) -> dict[str, float]:
    data = load_dataset(dataset_path)
    items = list(data.items())[offset:]
    if limit is not None:
        items = items[:limit]

    results: list[EvaluationResult] = []
    serializable: list[dict[str, Any]] = []

    for example_id, raw_example in items:
        try:
            builder_output = prepare_builder_output(example_id, raw_example)
            node = run_search(builder_output, algorithm=algorithm)
            predicted_plan = format_plan(builder_output, node)
            result = evaluate_prediction(
                example_id,
                raw_example,
                predicted_plan,
                include_golden=include_golden,
            )
        except Exception as exc:
            result = EvaluationResult(
                example_id=example_id,
                success=False,
                exact_match=False,
                predicted_plan="",
                error=str(exc),
            )
        results.append(result)
        row = {
            "example_id": result.example_id,
            "success": result.success,
            "exact_match": result.exact_match,
            "predicted_plan": result.predicted_plan,
            "error": result.error,
        }
        if include_golden:
            row["golden_plan"] = result.golden_plan
        serializable.append(row)

    metrics = summarize(results)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"metrics": metrics, "results": serializable}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PlanForge-guided trip planning runner.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--algorithm", choices=["bfs", "dfs", "astar"], default="bfs")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "trip_planning_predictions.json")
    parser.add_argument("--include-golden", action="store_true")
    args = parser.parse_args()

    metrics = run_pipeline(
        dataset_path=args.dataset,
        limit=args.limit,
        offset=args.offset,
        algorithm=args.algorithm,
        output_path=args.output,
        include_golden=args.include_golden,
    )
    print(f"Evaluated {int(metrics['count'])} examples")
    print(f"SR: {metrics['SR']:.4f}")
    print(f"EM: {metrics['EM']:.4f}")


if __name__ == "__main__":
    main()
