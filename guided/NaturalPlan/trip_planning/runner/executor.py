from __future__ import annotations

from typing import Any, Dict

from architect.integration_contract import BuilderOutput, validate_builder_output
from builder.nl2state_agent import parse_trip_example
from builder.refinement_loop import build_with_refinement


def prepare_builder_output(example_id: str, raw_example: Dict[str, Any]) -> BuilderOutput:
    problem = parse_trip_example(example_id, raw_example)
    output = build_with_refinement(problem)
    validate_builder_output(output)
    return output

