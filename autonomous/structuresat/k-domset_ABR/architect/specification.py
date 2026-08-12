"""Frozen Architect specification for the k-domset PlanForge run.

The Architect inspected only the first 20 development instances and selected a
complete, domain-neutral SAT method. The Builder implements this specification,
and the Runner freezes it only after the development split reaches 100% success.
"""

from __future__ import annotations

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
    name="complete_dpll_dimacs_k_domset",
    version="1.0",
    state_representation=(
        "Each k-domset instance is represented only as a DIMACS CNF formula: "
        "clauses are immutable tuples of integer literals, and search states "
        "are variable-to-boolean assignments. The representation intentionally "
        "does not depend on file names, label folders, or graph-generator "
        "metadata."
    ),
    algorithm=(
        "Use complete DPLL search with repeated clause simplification, unit "
        "propagation, pure literal elimination, contradiction detection, and "
        "chronological backtracking. This is a general exact SAT solver chosen "
        "from the development instances because their CNF sizes are small "
        "enough for complete search, and because exact search avoids "
        "dataset-label or domain-specific shortcuts."
    ),
    branching=(
        "At each branch, inspect the shortest unresolved clauses, choose an "
        "unassigned variable with the highest occurrence count there, and try "
        "the locally majority polarity first. If that branch fails, backtrack "
        "to the opposite polarity."
    ),
    validation=(
        "A SAT prediction succeeds only when the returned assignment satisfies "
        "every parsed clause. An UNSAT prediction succeeds when complete search "
        "exhausts all branches. SAT/UNSAT accuracy is computed against labels "
        "derived from sat/ or unsat/ dataset directories after inference."
    ),
)
