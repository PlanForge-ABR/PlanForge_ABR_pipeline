"""Executable PlanForge-guided pipeline for NaturalPlan meeting_planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from builder.nl2state_agent import nl2state_agent
from builder.refinement_loop import refinement_loop
from runner.evaluation import constraint_satisfied, exact_match, summarize
from runner.executor import load_planner
from runner.search_runner import plan_lines, run_search


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "NaturalPlan" / "data" / "meeting_planning.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "meeting_planning_predictions.json"


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def solve_example(example_id: str, example: dict) -> tuple[list[str], bool]:
    problem = nl2state_agent(example_id, example)
    bundle = load_planner(refinement_loop(problem))
    final_state = run_search(bundle)
    predicted = plan_lines(bundle, final_state)
    return predicted, constraint_satisfied(problem, predicted)


def run_pipeline(dataset_path: Path, output_path: Path, limit: int | None = None) -> dict:
    data = load_dataset(dataset_path)
    rows = []
    predictions = {}
    for index, (example_id, example) in enumerate(data.items()):
        if limit is not None and index >= limit:
            break
        predicted_plan, success = solve_example(example_id, example)
        golden_plan = example.get("golden_plan", [])
        em = exact_match(predicted_plan, golden_plan)
        rows.append(
            {
                "example_id": example_id,
                "success": success,
                "exact_match": em,
                "predicted_plan": predicted_plan,
            }
        )
        predictions[example_id] = {
            "predicted_plan": predicted_plan,
            "success": success,
            "exact_match": em,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    metrics = summarize(rows)
    metrics["output_path"] = str(output_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metrics = run_pipeline(args.dataset, args.output, args.limit)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
