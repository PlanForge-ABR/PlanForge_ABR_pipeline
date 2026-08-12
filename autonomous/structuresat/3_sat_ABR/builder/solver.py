"""Complete DPLL SAT solver used by the Builder implementation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from builder.cnf import CNFFormula, assignment_satisfies


@dataclass(frozen=True)
class SolveResult:
    status: str
    prediction: str
    assignment: dict[int, bool] | None
    runtime_seconds: float
    decisions: int


@dataclass
class _SearchStats:
    decisions: int = 0


def solve(formula: CNFFormula) -> SolveResult:
    start = perf_counter()
    stats = _SearchStats()
    assignment = _dpll(formula.clauses, {}, formula.num_vars, stats)
    runtime = perf_counter() - start

    if assignment is None:
        return SolveResult(
            status="SUCCESS",
            prediction="UNSAT",
            assignment=None,
            runtime_seconds=runtime,
            decisions=stats.decisions,
        )

    completed = {var: assignment.get(var, False) for var in range(1, formula.num_vars + 1)}
    if not assignment_satisfies(formula.clauses, completed):
        return SolveResult(
            status="FAILURE",
            prediction="SAT",
            assignment=completed,
            runtime_seconds=runtime,
            decisions=stats.decisions,
        )
    return SolveResult(
        status="SUCCESS",
        prediction="SAT",
        assignment=completed,
        runtime_seconds=runtime,
        decisions=stats.decisions,
    )


def _dpll(
    clauses: tuple[tuple[int, ...], ...],
    assignment: dict[int, bool],
    num_vars: int,
    stats: _SearchStats,
) -> dict[int, bool] | None:
    reduced, propagated = _propagate(clauses, assignment)
    if reduced is None:
        return None
    if not reduced:
        return propagated

    variable, preferred_value = _choose_branch(reduced, propagated, num_vars)
    if variable is None:
        return propagated

    for value in (preferred_value, not preferred_value):
        stats.decisions += 1
        branched = dict(propagated)
        branched[variable] = value
        result = _dpll(reduced, branched, num_vars, stats)
        if result is not None:
            return result
    return None


def _propagate(
    clauses: tuple[tuple[int, ...], ...],
    assignment: dict[int, bool],
) -> tuple[tuple[tuple[int, ...], ...] | None, dict[int, bool]]:
    current = dict(assignment)

    while True:
        reduced = _simplify(clauses, current)
        if reduced is None:
            return None, current
        if not reduced:
            return reduced, current

        implied: dict[int, bool] = {}
        for clause in reduced:
            if len(clause) == 1:
                literal = clause[0]
                implied[abs(literal)] = literal > 0

        polarity: dict[int, int] = {}
        for clause in reduced:
            for literal in clause:
                var = abs(literal)
                if var in current:
                    continue
                bit = 1 if literal > 0 else 2
                polarity[var] = polarity.get(var, 0) | bit
        for var, mask in polarity.items():
            if mask == 1:
                implied[var] = True
            elif mask == 2:
                implied[var] = False

        if not implied:
            return reduced, current

        changed = False
        for var, value in implied.items():
            existing = current.get(var)
            if existing is not None and existing != value:
                return None, current
            if existing is None:
                current[var] = value
                changed = True
        if not changed:
            return reduced, current


def _simplify(
    clauses: tuple[tuple[int, ...], ...],
    assignment: dict[int, bool],
) -> tuple[tuple[int, ...], ...] | None:
    reduced: list[tuple[int, ...]] = []
    for clause in clauses:
        unresolved: list[int] = []
        satisfied = False
        for literal in clause:
            value = assignment.get(abs(literal))
            if value is None:
                unresolved.append(literal)
            elif value == (literal > 0):
                satisfied = True
                break
        if satisfied:
            continue
        if not unresolved:
            return None
        reduced.append(tuple(unresolved))
    return tuple(reduced)


def _choose_branch(
    clauses: tuple[tuple[int, ...], ...],
    assignment: dict[int, bool],
    num_vars: int,
) -> tuple[int | None, bool]:
    if not clauses:
        return None, True

    shortest = min(len(clause) for clause in clauses)
    positive: dict[int, int] = {}
    negative: dict[int, int] = {}
    for clause in clauses:
        if len(clause) != shortest:
            continue
        for literal in clause:
            var = abs(literal)
            if var in assignment:
                continue
            if literal > 0:
                positive[var] = positive.get(var, 0) + 1
            else:
                negative[var] = negative.get(var, 0) + 1

    best_var: int | None = None
    best_score = -1
    best_balance = -1
    for var in range(1, num_vars + 1):
        if var in assignment:
            continue
        pos = positive.get(var, 0)
        neg = negative.get(var, 0)
        score = pos + neg
        balance = max(pos, neg)
        if score > best_score or (score == best_score and balance > best_balance):
            best_var = var
            best_score = score
            best_balance = balance

    if best_var is None:
        return None, True
    return best_var, positive.get(best_var, 0) >= negative.get(best_var, 0)
