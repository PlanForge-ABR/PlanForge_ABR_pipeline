from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from architect.state_schema import TripProblem, TripState


@dataclass(frozen=True)
class BuilderOutput:
    problem: TripProblem
    initial_state: TripState
    goal_test: Callable[[TripState], bool]
    successor_fn: Callable[[TripState], Iterable[TripState]]


def validate_builder_output(output: BuilderOutput) -> None:
    if not isinstance(output.problem, TripProblem):
        raise TypeError("Builder output must include a TripProblem.")
    if not isinstance(output.initial_state, TripState):
        raise TypeError("Builder output must include a TripState initial_state.")
    if not callable(output.goal_test):
        raise TypeError("Builder output goal_test must be callable.")
    if not callable(output.successor_fn):
        raise TypeError("Builder output successor_fn must be callable.")

