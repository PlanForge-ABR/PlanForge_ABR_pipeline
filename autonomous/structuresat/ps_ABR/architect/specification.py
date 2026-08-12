"""Frozen Architect specification for the PS PlanForge SAT run.

The Architect inspected only the first 20 development instances and selected a
complete, domain-neutral CNF SAT methodology. The Builder implements this
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
    name="complete_dpll_dimacs_ps",
    version="1.0",
    state_representation=(
        "Each PS SAT instance is parsed as a DIMACS CNF formula stored as "
        "immutable tuples of integer literals. A search state is a "
        "variable-to-boolean mapping; positive literals require True and "
        "negative literals require False."
    ),
    algorithm=(
        "Use complete DPLL search with repeated clause simplification, unit "
        "propagation, pure literal elimination, contradiction detection, and "
        "chronological backtracking. The method is a general SAT solver and "
        "does not infer predictions from file names, ordering, or label "
        "directories."
    ),
    branching=(
        "Branch on an unassigned variable with the largest occurrence count "
        "among the shortest unresolved clauses. Try the locally dominant "
        "polarity first, then backtrack to the opposite polarity if needed."
    ),
    validation=(
        "A SAT prediction succeeds only when the returned assignment satisfies "
        "every clause. An UNSAT prediction succeeds when complete search "
        "exhausts all branches. SAT/UNSAT accuracy is measured against the "
        "dataset label directory only after inference."
    ),
)

