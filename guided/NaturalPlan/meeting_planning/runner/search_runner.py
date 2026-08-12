"""Run deterministic search for the best meeting schedule."""

from __future__ import annotations

from architect.integration_contract import PlannerBundle
from architect.search_library import exhaustive_best
from architect.state_schema import MeetingState


def run_search(bundle: PlannerBundle) -> MeetingState:
    planner = bundle
    return exhaustive_best(planner.initial_state, planner.successor_fn, planner.score_fn)


def plan_lines(bundle: PlannerBundle, state: MeetingState) -> list[str]:
    return bundle.formatter(state)

