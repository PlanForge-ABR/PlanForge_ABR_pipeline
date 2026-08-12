from __future__ import annotations

from architect.integration_contract import BuilderOutput, validate_builder_output
from architect.state_schema import TripProblem
from builder.function_generator import generate_functions
from builder.test_generator import run_generated_tests


def build_with_refinement(problem: TripProblem, budget: int = 3) -> BuilderOutput:
    last_errors: list[str] = []
    for _ in range(max(1, budget)):
        output = generate_functions(problem)
        validate_builder_output(output)
        last_errors = run_generated_tests(output)
        if not last_errors:
            return output
    raise RuntimeError(f"{problem.example_id}: generated functions failed tests: {last_errors}")

