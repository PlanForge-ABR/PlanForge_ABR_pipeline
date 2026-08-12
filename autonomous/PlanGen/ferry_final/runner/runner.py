"""Deterministic execution and reporting pipeline for ferry."""

import json
from pathlib import Path
from typing import Any, Dict, List

from builder.parser import parse_instance
from builder.planner import construct_plan, goals_hold, simulate_plan


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_dataset(split: str = "test", write_outputs: bool = True) -> Dict[str, Any]:
    dataset_path = PROJECT_ROOT / split / "ferry.json"
    data = json.loads(dataset_path.read_text())
    results: List[Dict[str, Any]] = []

    correct_existence = 0
    checked = 0

    for item in data:
        state, goals = parse_instance(item["context"], item["inputs"])
        result = construct_plan(state, goals)
        predicted = "yes" if result.exists else "no"
        row = {
            "id": item["id"],
            "plan_existence": predicted,
            "predicted_plan": result.plan if result.exists else [],
            "reason": result.reason,
        }

        if split == "train":
            answer = item.get("answer")
            row["answer"] = answer
            row["existence_correct"] = predicted == answer
            row["plan_validates"] = _validates(state, goals, result.plan) if result.exists else predicted == answer
            checked += 1
            correct_existence += int(predicted == answer)

        results.append(row)

    summary = {
        "domain": "ferry",
        "split": split,
        "instances": len(results),
        "results": results,
    }
    if split == "train" and checked:
        summary["plan_existence_accuracy"] = correct_existence / checked
        summary["all_train_existence_correct"] = correct_existence == checked
        summary["all_train_plans_valid"] = all(r.get("plan_validates", True) for r in results)

    if write_outputs:
        out_dir = DOMAIN_ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{split}_results.json").write_text(json.dumps(summary, indent=2))

    return summary


def _validates(state, goals, plan) -> bool:
    try:
        return goals_hold(simulate_plan(state, plan), goals)
    except Exception:
        return False
