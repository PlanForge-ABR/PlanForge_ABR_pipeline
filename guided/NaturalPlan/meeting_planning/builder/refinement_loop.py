"""PlanForge-style generate-test-refine loop."""

from __future__ import annotations

from architect.integration_contract import PlannerBundle, validate_bundle
from architect.state_schema import MeetingProblem
from builder.function_generator import generate_functions
from builder.test_generator import generate_tests


def refinement_loop(problem: MeetingProblem, max_iterations: int = 3) -> PlannerBundle:
    errors: list[Exception] = []
    for _ in range(max_iterations):
        bundle = validate_bundle(generate_functions(problem))
        try:
            for test in generate_tests(problem, bundle):
                test()
            return bundle
        except AssertionError as exc:
            errors.append(exc)
    raise RuntimeError(f"generated planner failed tests after {max_iterations} iterations: {errors}")

