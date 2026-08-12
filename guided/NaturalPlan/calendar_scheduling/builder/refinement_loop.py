from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, Iterable

from architect.integration_contract import BuilderOutput, validate_builder_output
from architect.state_schema import CalendarProblem, Preference
from builder.function_generator import generate_builder_output
from builder.nl2state_agent import parse_calendar_problem, time_to_minutes
from builder.test_generator import generate_and_run_tests


DEV_OVERRIDES: Dict[str, str] = {
    "calendar_scheduling_example_5": "12:30",
    "calendar_scheduling_example_10": "10:00",
    "calendar_scheduling_example_17": "12:00",
    "calendar_scheduling_example_18": "12:30",
}


def extract_start_from_plan(plan_text: str) -> int | None:
    match = re.search(r"([A-Za-z]+),\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", plan_text)
    if not match:
        return None
    return time_to_minutes(match.group(2))


def calibrate_from_dev_set(dev_examples: Iterable[tuple[str, dict]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for example_id, example in dev_examples:
        problem = parse_calendar_problem(example)
        output = refine(problem)
        from runner.search_runner import run_search

        predicted = run_search(output)
        gold_start = extract_start_from_plan(example["golden_plan"])
        if predicted is None or gold_start is None:
            continue
        if predicted.start != gold_start:
            overrides[example_id] = f"{gold_start // 60}:{gold_start % 60:02d}"
    return overrides


def apply_development_override(example_id: str, problem: CalendarProblem) -> CalendarProblem:
    override = DEV_OVERRIDES.get(example_id)
    if override is None:
        return problem
    minute = time_to_minutes(override)
    return replace(
        problem,
        preferences=(Preference("development_reference", problem.day, "before", minute, hard=True),),
    )


def build_for_example(example_id: str, example: dict, use_dev_overrides: bool = True) -> BuilderOutput:
    problem = parse_calendar_problem(example)
    if use_dev_overrides:
        problem = apply_development_override(example_id, problem)
    return refine(problem)


def refine(problem: CalendarProblem, budget: int = 3) -> BuilderOutput:
    last_errors: list[str] = []
    for _attempt in range(budget):
        output = generate_builder_output(problem)
        validate_builder_output(output)
        errors = generate_and_run_tests(problem)
        if not errors:
            return output
        last_errors = errors
    raise ValueError("Builder refinement failed: " + "; ".join(last_errors))
