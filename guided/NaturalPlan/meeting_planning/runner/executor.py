"""Load builder output for execution."""

from __future__ import annotations

from architect.integration_contract import PlannerBundle, validate_bundle


def load_planner(bundle: PlannerBundle) -> PlannerBundle:
    return validate_bundle(bundle)

