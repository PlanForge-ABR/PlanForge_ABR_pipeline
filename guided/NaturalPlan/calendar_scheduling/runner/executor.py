from __future__ import annotations

from architect.integration_contract import BuilderOutput, validate_builder_output


def load_builder_output(output: BuilderOutput) -> BuilderOutput:
    validate_builder_output(output)
    return output

