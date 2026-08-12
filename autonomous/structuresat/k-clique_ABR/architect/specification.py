"""Frozen Architect specification for the k-clique PlanForge run.

The Architect selected a complete, domain-neutral SAT methodology after
inspecting only the first 20 development instances. The Builder implements this
specification, and the Runner freezes it after development success.
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
    name="complete_dpll_dimacs_k_clique",
    version="1.0",
    state_representation=(
        "Each k-clique SAT instance is read as a DIMACS CNF formula and stored "
        "as immutable tuples of integer literals. Positive literals represent "
        "True assignments, negative literals represent False assignments, and "
        "a search state is a variable-to-boolean mapping."
    ),
    algorithm=(
        "Use complete DPLL search with repeated clause simplification, unit "
        "propagation, pure literal elimination, contradiction detection, and "
        "chronological backtracking. The method is a general CNF SAT solver and "
        "does not infer predictions from file names or label directories."
    ),
    branching=(
        "Choose an unassigned variable with the largest occurrence count among "
        "currently shortest unresolved clauses. Try the polarity with the "
        "higher local occurrence count first, then backtrack to the opposite "
        "polarity if needed."
    ),
    validation=(
        "A SAT prediction succeeds only if the returned assignment satisfies "
        "every clause. An UNSAT prediction succeeds when complete search "
        "exhausts all branches. SAT/UNSAT accuracy is measured against the "
        "dataset label directory after inference."
    ),
)
