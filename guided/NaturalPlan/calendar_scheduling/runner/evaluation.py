from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from architect.state_schema import CalendarAction, CalendarProblem
from builder.function_generator import slot_is_valid
from builder.nl2state_agent import minutes_to_time, time_to_minutes


@dataclass(frozen=True)
class EvaluationRecord:
    example_id: str
    predicted_plan: str
    golden_plan: str
    success: bool
    exact_match: bool


def format_plan(action: CalendarAction | None) -> str:
    if action is None:
        return "⊥"
    return (
        f"Here is the proposed time: {action.day}, "
        f"{minutes_to_time(action.start)} - {minutes_to_time(action.end)} "
    )


def normalize_plan(plan: str) -> str:
    return " ".join(plan.replace("SOLUTION:", "").strip().split())


def parse_plan_slot(plan: str) -> tuple[str, int, int] | None:
    match = re.search(r"([A-Za-z]+),\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", plan)
    if not match:
        return None
    return match.group(1), time_to_minutes(match.group(2)), time_to_minutes(match.group(3))


def satisfies_constraints(problem: CalendarProblem, predicted_plan: str) -> bool:
    slot = parse_plan_slot(predicted_plan)
    if slot is None:
        return False
    day, start, end = slot
    if day not in problem.days:
        return False
    return slot_is_valid(problem, start, end, day)


def evaluate_one(
    example_id: str,
    problem: CalendarProblem,
    action: CalendarAction | None,
    golden_plan: str,
) -> EvaluationRecord:
    predicted_plan = format_plan(action)
    return EvaluationRecord(
        example_id=example_id,
        predicted_plan=predicted_plan,
        golden_plan=golden_plan,
        success=satisfies_constraints(problem, predicted_plan),
        exact_match=normalize_plan(predicted_plan) == normalize_plan(golden_plan),
    )


def aggregate(records: Iterable[EvaluationRecord]) -> dict:
    rows: List[EvaluationRecord] = list(records)
    total = len(rows)
    if total == 0:
        return {"total": 0, "SR": 0.0, "EM": 0.0}
    return {
        "total": total,
        "SR": sum(record.success for record in rows) / total,
        "EM": sum(record.exact_match for record in rows) / total,
    }
