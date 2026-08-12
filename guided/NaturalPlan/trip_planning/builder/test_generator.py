from __future__ import annotations

from architect.integration_contract import BuilderOutput
from architect.state_schema import TripProblem, TripState


def run_generated_tests(output: BuilderOutput) -> list[str]:
    problem = output.problem
    errors: list[str] = []
    initial = output.initial_state

    if initial.current_city is not None or initial.visited_cities:
        errors.append("Initial trip state must not have visited any city.")
    if initial.remaining_cities != frozenset(problem.cities):
        errors.append("Initial remaining_cities must contain every problem city.")

    first_successors = list(output.successor_fn(initial))
    if not first_successors:
        errors.append("Successor function must allow at least one starting city.")

    for state in first_successors:
        _check_state(problem, state, errors)
        second_successors = list(output.successor_fn(state))
        for child in second_successors:
            _check_state(problem, child, errors)
            if len(child.visited_cities) != len(set(child.visited_cities)):
                errors.append("A successor revisits a city.")
            if (state.current_city, child.current_city) not in problem.flights:
                errors.append("A successor uses a non-direct flight.")

    return errors


def _check_state(problem: TripProblem, state: TripState, errors: list[str]) -> None:
    if len(state.visited_cities) + len(state.remaining_cities) != problem.num_cities:
        errors.append("Visited and remaining city counts do not match num_cities.")
    if any(city in state.remaining_cities for city in state.visited_cities):
        errors.append("A city appears in both visited and remaining sets.")
    if state.total_days_used > problem.total_days:
        errors.append("State exceeds total trip duration.")

