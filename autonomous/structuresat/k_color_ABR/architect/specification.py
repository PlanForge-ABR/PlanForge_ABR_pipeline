"""Frozen Architect specification for the k-color PlanForge run.

The Architect selected a complete, domain-neutral SAT methodology after
inspecting only development-set structure and sizes. The Builder implements
this specification, and the Runner freezes it after all 20 development
instances pass.
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
    name="complete_dpll_dimacs_k_color",
    version="1.0",
    state_representation=(
        "Each DIMACS CNF instance is represented as immutable tuples of integer "
        "literals. A positive literal means the variable is True, a negative "
        "literal means False, and a solver state is a variable-to-boolean map."
    ),
    algorithm=(
        "Use complete DPLL search with repeated clause simplification, unit "
        "propagation, pure literal elimination, contradiction detection, and "
        "backtracking. The method is general for SAT CNF and does not infer the "
        "answer from dataset path names."
    ),
    branching=(
        "Choose an unassigned variable with the largest occurrence count in the "
        "currently shortest unresolved clauses. Try the polarity that appears "
        "more often first, then backtrack to the opposite polarity if needed."
    ),
    validation=(
        "A SAT prediction is successful only when the returned assignment "
        "satisfies every clause. An UNSAT prediction is successful when complete "
        "search exhausts all branches without a satisfying assignment. "
        "Prediction accuracy is measured against sat/unsat label directories."
    ),
)
