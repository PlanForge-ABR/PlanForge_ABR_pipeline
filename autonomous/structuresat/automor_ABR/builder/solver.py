"""Self-contained complete SAT solvers for automor CNF instances."""

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
    csp_result = _solve_permutation_csp(formula)
    if csp_result is not None:
        csp_assignment, csp_decisions = csp_result
        runtime = perf_counter() - start
        if csp_assignment:
            completed = {
                variable: csp_assignment.get(variable, False)
                for variable in range(1, formula.num_vars + 1)
            }
            status = "SUCCESS" if assignment_satisfies(formula.clauses, completed) else "FAILURE"
            return SolveResult(
                status=status,
                prediction="SAT",
                assignment=completed,
                runtime_seconds=runtime,
                decisions=csp_decisions,
            )
        return SolveResult(
            status="SUCCESS",
            prediction="UNSAT",
            assignment=None,
            runtime_seconds=runtime,
            decisions=csp_decisions,
        )

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

    completed = {
        variable: assignment.get(variable, False)
        for variable in range(1, formula.num_vars + 1)
    }
    status = "SUCCESS" if assignment_satisfies(formula.clauses, completed) else "FAILURE"
    return SolveResult(
        status=status,
        prediction="SAT",
        assignment=completed,
        runtime_seconds=runtime,
        decisions=stats.decisions,
    )


def _solve_permutation_csp(formula: CNFFormula) -> tuple[dict[int, bool] | None, int] | None:
    n = _square_root(formula.num_vars)
    if n is None:
        return None

    expected_rows = {tuple(range(row * n + 1, row * n + n + 1)) for row in range(n)}
    expected_cols = {
        tuple(row * n + col + 1 for row in range(n))
        for col in range(n)
    }
    positive_exactly_one = {
        clause for clause in formula.clauses if len(clause) == n and all(lit > 0 for lit in clause)
    }
    if not expected_rows.issubset(positive_exactly_one):
        return None
    if not expected_cols.issubset(positive_exactly_one):
        return None

    incompatible: list[set[int]] = [set() for _ in range(formula.num_vars + 1)]
    for clause in formula.clauses:
        if len(clause) == n and all(lit > 0 for lit in clause):
            continue
        if len(clause) != 2 or clause[0] >= 0 or clause[1] >= 0:
            return None
        a = -clause[0]
        b = -clause[1]
        if a < 1 or b < 1 or a > formula.num_vars or b > formula.num_vars:
            return None
        incompatible[a].add(b)
        incompatible[b].add(a)

    decisions = 0
    memo: set[tuple[tuple[int, ...], int]] = set()

    def var_for(row: int, col: int) -> int:
        return row * n + col + 1

    def compatible_with_assigned(row: int, col: int, assignment: tuple[int, ...]) -> bool:
        variable = var_for(row, col)
        for other_row, other_col in enumerate(assignment):
            if other_col < 0:
                continue
            if var_for(other_row, other_col) in incompatible[variable]:
                return False
        return True

    def feasible_cols(row: int, assignment: tuple[int, ...], used_mask: int) -> list[int]:
        cols: list[int] = []
        for col in range(n):
            if used_mask & (1 << col):
                continue
            if compatible_with_assigned(row, col, assignment):
                cols.append(col)
        return cols

    def has_forward_support(assignment: tuple[int, ...], used_mask: int) -> bool:
        for row, col in enumerate(assignment):
            if col < 0 and not feasible_cols(row, assignment, used_mask):
                return False
        return True

    def order_columns(row: int, cols: list[int], assignment: tuple[int, ...], used_mask: int) -> list[int]:
        scored: list[tuple[int, int]] = []
        for col in cols:
            variable = var_for(row, col)
            support = 0
            next_mask = used_mask | (1 << col)
            for other_row, other_col in enumerate(assignment):
                if other_col >= 0 or other_row == row:
                    continue
                for other_col_candidate in range(n):
                    if next_mask & (1 << other_col_candidate):
                        continue
                    other_var = var_for(other_row, other_col_candidate)
                    if other_var not in incompatible[variable]:
                        support += 1
            scored.append((-support, col))
        return [col for _, col in sorted(scored)]

    def search(assignment: tuple[int, ...], used_mask: int) -> tuple[int, ...] | None:
        nonlocal decisions
        if all(col >= 0 for col in assignment):
            return assignment

        key = (assignment, used_mask)
        if key in memo:
            return None

        best_row: int | None = None
        best_cols: list[int] = []
        for row, col in enumerate(assignment):
            if col >= 0:
                continue
            cols = feasible_cols(row, assignment, used_mask)
            if not cols:
                memo.add(key)
                return None
            if best_row is None or len(cols) < len(best_cols):
                best_row = row
                best_cols = cols

        assert best_row is not None
        for col in order_columns(best_row, best_cols, assignment, used_mask):
            decisions += 1
            next_assignment = list(assignment)
            next_assignment[best_row] = col
            next_tuple = tuple(next_assignment)
            next_mask = used_mask | (1 << col)
            if not has_forward_support(next_tuple, next_mask):
                continue
            result = search(next_tuple, next_mask)
            if result is not None:
                return result

        memo.add(key)
        return None

    solution = search(tuple(-1 for _ in range(n)), 0)
    if solution is None:
        return {}, decisions

    assignment = {variable: False for variable in range(1, formula.num_vars + 1)}
    for row, col in enumerate(solution):
        assignment[var_for(row, col)] = True
    return assignment, decisions


def _square_root(value: int) -> int | None:
    if value < 1:
        return None
    root = int(value**0.5)
    if root * root == value:
        return root
    if (root + 1) * (root + 1) == value:
        return root + 1
    return None


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
                variable = abs(literal)
                if variable in current:
                    continue
                polarity[variable] = polarity.get(variable, 0) | (1 if literal > 0 else 2)

        for variable, mask in polarity.items():
            if mask == 1:
                implied[variable] = True
            elif mask == 2:
                implied[variable] = False

        if not implied:
            return reduced, current

        changed = False
        for variable, value in implied.items():
            existing = current.get(variable)
            if existing is not None and existing != value:
                return None, current
            if existing is None:
                current[variable] = value
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
        for literal in clause:
            value = assignment.get(abs(literal))
            if value is None:
                unresolved.append(literal)
            elif value == (literal > 0):
                break
        else:
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
            variable = abs(literal)
            if variable in assignment:
                continue
            if literal > 0:
                positive[variable] = positive.get(variable, 0) + 1
            else:
                negative[variable] = negative.get(variable, 0) + 1

    best_variable: int | None = None
    best_score = -1
    best_balance = -1
    for variable in range(1, num_vars + 1):
        if variable in assignment:
            continue
        pos = positive.get(variable, 0)
        neg = negative.get(variable, 0)
        score = pos + neg
        balance = max(pos, neg)
        if score > best_score or (score == best_score and balance > best_balance):
            best_variable = variable
            best_score = score
            best_balance = balance

    if best_variable is None:
        return None, True
    return best_variable, positive.get(best_variable, 0) >= negative.get(best_variable, 0)
