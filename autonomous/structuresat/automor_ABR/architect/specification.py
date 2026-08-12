"""Architect methodology for the automor PlanForge SAT run.

The Architect inspected only the first 20 development instances to choose a
general complete SAT methodology. The selected method is intentionally
domain-neutral at the CNF level: it parses DIMACS, applies sound formula
simplifications, and uses complete backtracking search when needed.
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
    freeze_rule: str


FROZEN_SPEC = MethodologySpec(
    name="complete_permutation_csp_with_dpll_fallback_automor",
    version="2.0",
    state_representation=(
        "A DIMACS CNF instance is represented as tuples of integer literals. "
        "Assignments map 1-based variable ids to booleans; positive literals "
        "require True and negative literals require False."
    ),
    algorithm=(
        "When the CNF has a square exact-one permutation-matrix encoding, "
        "compile it into a finite-domain permutation CSP and solve it with "
        "complete backtracking, forward checking, and memoization over partial "
        "assignments. This is sound because every CSP assignment maps directly "
        "to the DIMACS variables and all binary negative clauses are enforced. "
        "For non-matching CNFs, fall back to complete DPLL with repeated clause "
        "simplification, unit propagation, pure literal elimination, "
        "contradiction detection, and recursive backtracking."
    ),
    branching=(
        "For permutation CSPs, choose the unassigned row with the fewest "
        "currently feasible columns and try columns with the strongest "
        "compatibility support first. For fallback DPLL, branch on an "
        "unassigned variable that appears most often in the currently shortest "
        "unresolved clauses and try the more frequent polarity first."
    ),
    validation=(
        "SAT predictions must include an assignment satisfying every clause. "
        "UNSAT predictions are accepted only when complete search returns no "
        "satisfying assignment. Accuracy is measured separately against the "
        "dataset label directory."
    ),
    freeze_rule=(
        "Freeze the Architect specification, Builder code, and Runner pipeline "
        "only after all 20 development instances return SUCCESS."
    ),
)
