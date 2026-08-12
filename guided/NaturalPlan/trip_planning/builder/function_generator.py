from __future__ import annotations

from typing import Iterable

from architect.integration_contract import BuilderOutput
from architect.state_schema import Segment, TripProblem, TripState


def generate_functions(problem: TripProblem) -> BuilderOutput:
    initial_state = TripState(
        current_city=None,
        visited_cities=tuple(),
        remaining_cities=frozenset(problem.cities),
        total_days_used=0,
        segments=tuple(),
    )

    def goal_test(state: TripState) -> bool:
        if state.remaining_cities:
            return False
        if state.total_days_used != problem.total_days:
            return False
        return all(_constraint_satisfied(state, constraint.city, constraint.start_day, constraint.end_day)
                   for constraint in problem.constraints)

    def successor_fn(state: TripState) -> Iterable[TripState]:
        if not state.remaining_cities:
            return []

        successors: list[TripState] = []
        for city in problem.cities:
            if city not in state.remaining_cities:
                continue
            if state.current_city is not None and (state.current_city, city) not in problem.flights:
                continue

            start_day = 1 if state.current_city is None else state.total_days_used
            end_day = start_day + problem.durations[city] - 1
            if end_day > problem.total_days:
                continue

            segment = Segment(
                city=city,
                start_day=start_day,
                end_day=end_day,
                duration=problem.durations[city],
            )
            next_state = TripState(
                current_city=city,
                visited_cities=state.visited_cities + (city,),
                remaining_cities=frozenset(c for c in state.remaining_cities if c != city),
                total_days_used=end_day,
                segments=state.segments + (segment,),
            )
            if _can_still_satisfy_constraints(problem, next_state):
                successors.append(next_state)
        return successors

    return BuilderOutput(
        problem=problem,
        initial_state=initial_state,
        goal_test=goal_test,
        successor_fn=successor_fn,
    )


def _constraint_satisfied(state: TripState, city: str, start_day: int, end_day: int) -> bool:
    return any(
        segment.city == city and segment.start_day <= start_day and segment.end_day >= end_day
        for segment in state.segments
    )


def _can_still_satisfy_constraints(problem: TripProblem, state: TripState) -> bool:
    for constraint in problem.constraints:
        city = constraint.city
        if city in state.visited_cities:
            if not _constraint_satisfied(state, city, constraint.start_day, constraint.end_day):
                return False
        elif constraint.end_day < state.total_days_used:
            return False
    return True

