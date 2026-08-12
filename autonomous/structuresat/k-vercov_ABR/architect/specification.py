"""Frozen Architect specification for the k-vercov PlanForge run.

The Architect inspected only the first 20 development instances, compared the
observed DIMACS size and structure, and selected a complete, domain-neutral SAT
method. The Builder implements this specification, and the Runner freezes it
only after the development split reaches 100% success.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodologySpec:
    name: str
    version: str
    development_observation: str
    state_representation: str
    algorithm: str
    branching: str
    validation: str


FROZEN_SPEC = MethodologySpec(
    name="complete_dpll_dimacs_k_vercov",
    version="1.0",
    development_observation=(
        "The first 20 k-vercov development instances are ordinary DIMACS CNF "
        "formulas with modest variable counts and mixed filename suffixes "
        "(.cnf, _m_cnf, _n_cnf). The selected methodology treats all of them "
        "only as formulas and does not use filename patterns, generation "
        "suffixes, or label directories to infer satisfiability."
    ),
    state_representation=(
        "Each instance is represented as a DIMACS CNF formula: clauses are "
        "immutable tuples of integer literals, and search states are "
        "variable-to-boolean assignments."
    ),
    algorithm=(
        "Use complete DPLL search with repeated clause simplification, unit "
        "propagation, pure literal elimination, contradiction detection, and "
        "chronological backtracking. This exact method was selected because "
        "it is unbiased across SAT domains and the development formulas are "
        "small enough for complete search."
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
        "exhausts all branches. SAT/UNSAT accuracy is computed against dataset "
        "labels only after inference."
    ),
)
