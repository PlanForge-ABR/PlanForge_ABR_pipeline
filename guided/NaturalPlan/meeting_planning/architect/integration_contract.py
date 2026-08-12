"""Builder-to-runner contract validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .state_schema import MeetingState


@dataclass(frozen=True)
class PlannerBundle:
    initial_state: MeetingState
    goal_test: Callable[[MeetingState], bool]
    successor_fn: Callable[[MeetingState], list[MeetingState]]
    score_fn: Callable[[MeetingState], tuple]
    formatter: Callable[[MeetingState], list[str]]


def validate_bundle(bundle: PlannerBundle) -> PlannerBundle:
    if not isinstance(bundle.initial_state, MeetingState):
        raise TypeError("initial_state must be a MeetingState")
    for name in ("goal_test", "successor_fn", "score_fn", "formatter"):
        if not callable(getattr(bundle, name)):
            raise TypeError(f"{name} must be callable")
    return bundle

