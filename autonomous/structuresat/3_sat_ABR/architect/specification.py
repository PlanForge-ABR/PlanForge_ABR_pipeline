"""Frozen Architect specification for the 3-SAT PlanForge run.

The development procedure may revise this file before freeze. Once the first
20 development instances pass, the Runner records this exact specification in
the output metadata and uses the same Builder implementation for evaluation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodologySpec:
    name: str
    version: str
    state_representation: str
    algorithm: str
    branching: str
    validation: str


FROZEN_SPEC = MethodologySpec(
    name="complete_dpll_dimacs_3sat",
    version="1.0",
    state_representation=(
        "DIMACS CNF parsed as integer literals; positive literals mean True, "
        "negative literals mean False, and an assignment is a variable-to-bool map."
    ),
    algorithm=(
        "Complete recursive DPLL search with repeated unit propagation, pure "
        "literal elimination, conflict detection, and backtracking."
    ),
    branching=(
        "Choose an unassigned variable with the highest occurrence count among "
        "currently unresolved shortest clauses; try the majority polarity first."
    ),
    validation=(
        "SAT requires a full or partial assignment satisfying every clause. "
        "UNSAT is accepted only when the complete search proves no assignment."
    ),
)
