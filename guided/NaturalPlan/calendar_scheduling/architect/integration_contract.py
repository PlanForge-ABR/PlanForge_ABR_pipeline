from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

from architect.state_schema import CalendarAction, CalendarProblem, CalendarState


@dataclass(frozen=True)
class BuilderOutput:
    problem: CalendarProblem
    initial_state: CalendarState
    goal_test: Callable[[CalendarState], bool]
    successor_fn: Callable[[CalendarState], Iterable[Tuple[CalendarAction, CalendarState]]]


def validate_builder_output(output: BuilderOutput) -> None:
    if output.problem.duration <= 0:
        raise ValueError("Meeting duration must be positive.")
    if output.problem.work_start >= output.problem.work_end:
        raise ValueError("Work start must be before work end.")
    if not output.problem.people:
        raise ValueError("At least one participant is required.")
    for person in output.problem.people:
        if person not in output.problem.busy:
            raise ValueError(f"Missing busy schedule for {person}.")

