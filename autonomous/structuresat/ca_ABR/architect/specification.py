"""Frozen Architect specification for the CA PlanForge run.

The Architect selected a complete SAT procedure after inspecting the first
20 development instances through Runner feedback. Once those instances pass,
the Runner records this exact specification and evaluates the remaining 1000
instances without changing methodology or code.
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
    name="complete_dpll_dimacs_ca",
    version="1.0",
    state_representation=(
        "DIMACS CNF parsed into immutable tuples of integer literals; positive "
        "literals denote True, negative literals denote False, and solver state "
        "is a variable-to-boolean assignment plus unresolved clauses."
    ),
    algorithm=(
        "Complete recursive DPLL search with fixed-point unit propagation, pure "
        "literal elimination, clause simplification, conflict detection, and "
        "chronological backtracking. The method is domain-agnostic and does not "
        "use file names, label folders, or evaluation answers to choose outputs."
    ),
    branching=(
        "Select an unassigned variable from currently shortest unresolved "
        "clauses using highest occurrence count; try the majority observed "
        "polarity first and backtrack to the opposite polarity if needed."
    ),
    validation=(
        "A SAT prediction succeeds only when the returned assignment satisfies "
        "every parsed clause. An UNSAT prediction succeeds only when exhaustive "
        "DPLL search proves no satisfying assignment exists."
    ),
)
