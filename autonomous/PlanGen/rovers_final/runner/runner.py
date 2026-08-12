"""Deterministic execution and reporting pipeline for rovers."""

import json
from pathlib import Path
from typing import Any, Dict, List

from builder.parser import parse_instance
from builder.planner import construct_plan, goals_hold, simulate_plan


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_dataset(split: str = "test", write_outputs: bool = True) -> Dict[str, Any]:
    dataset_path = PROJECT_ROOT / split / "rovers.json"
    data = json.loads(dataset_path.read_text())
    results: List[Dict[str, Any]] = []
    correct_existence = 0
    valid_plans = 0

    for item in data:
        problem = parse_instance(item)
        result = construct_plan(problem)
        predicted = "yes" if result.exists else "no"
        row: Dict[str, Any] = {
            "id": item["id"],
            "plan_existence": predicted,
            "predicted_plan": result.plan if result.exists else [],
            "reason": result.reason,
        }

        if split == "train":
            answer = item.get("answer")
            row["answer"] = answer
            row["existence_correct"] = predicted == answer
            row["plan_validates"] = _validates(problem, result.plan, result.exists)
            correct_existence += int(predicted == answer)
            valid_plans += int(row["plan_validates"])

        results.append(row)

    summary: Dict[str, Any] = {
        "domain": "rovers",
        "split": split,
        "instances": len(results),
        "results": results,
    }
    if split == "train":
        count = len(results)
        summary["plan_existence_accuracy"] = correct_existence / count if count else 0.0
        summary["all_train_existence_correct"] = correct_existence == count
        summary["all_train_plans_valid"] = valid_plans == count

    if write_outputs:
        out_dir = DOMAIN_ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{split}_results.json").write_text(json.dumps(summary, indent=2))

    return summary


def _validates(problem, plan, exists: bool) -> bool:
    if not exists:
        return True
    try:
        return goals_hold(simulate_plan(problem, plan), problem.goal_facts)
    except Exception:
        return False
