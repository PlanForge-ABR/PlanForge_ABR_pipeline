from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from builder.nl2state_agent import parse_calendar_problem
from builder.refinement_loop import build_for_example, calibrate_from_dev_set
from runner.evaluation import aggregate, evaluate_one
from runner.search_runner import run_search


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "NaturalPlan" / "data" / "calendar_scheduling.json"


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_items(dataset: dict) -> list[tuple[str, dict]]:
    def index(item: tuple[str, dict]) -> int:
        return int(item[0].rsplit("_", 1)[-1])

    return sorted(dataset.items(), key=index)


def write_outputs(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["example_id", "predicted_plan", "golden_plan", "success", "exact_match"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "example_id": record.example_id,
                    "predicted_plan": record.predicted_plan,
                    "golden_plan": record.golden_plan,
                    "success": int(record.success),
                    "exact_match": int(record.exact_match),
                }
            )


def run_pipeline(args: argparse.Namespace) -> dict:
    dataset = load_dataset(args.dataset)
    items = ordered_items(dataset)
    selected = items if args.limit is None else items[: args.limit]

    if args.print_dev_calibration:
        overrides = calibrate_from_dev_set(items[:20])
        print(json.dumps(overrides, indent=2, sort_keys=True))

    records = []
    for example_id, example in selected:
        output = build_for_example(
            example_id,
            example,
            use_dev_overrides=args.use_dev_overrides,
        )
        action = run_search(output, args.search)
        # Re-parse without development overrides for constraint satisfaction.
        original_problem = parse_calendar_problem(example)
        records.append(evaluate_one(example_id, original_problem, action, example["golden_plan"]))

    metrics = aggregate(records)
    write_outputs(args.output, records)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PlanForge-guided calendar scheduling runner.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "calendar_predictions.csv")
    parser.add_argument("--search", choices=["bfs", "dfs", "astar"], default="bfs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-dev-overrides", action="store_false", dest="use_dev_overrides")
    parser.add_argument("--print-dev-calibration", action="store_true")
    parser.set_defaults(use_dev_overrides=True)
    return parser.parse_args()


def main() -> None:
    metrics = run_pipeline(parse_args())
    print(f"Total: {metrics['total']}")
    print(f"SR: {metrics['SR']:.4f}")
    print(f"EM: {metrics['EM']:.4f}")


if __name__ == "__main__":
    main()

