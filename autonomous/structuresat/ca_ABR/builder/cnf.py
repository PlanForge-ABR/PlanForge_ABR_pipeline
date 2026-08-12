"""DIMACS CNF parsing and assignment validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CNFFormula:
    path: Path
    num_vars: int
    clauses: tuple[tuple[int, ...], ...]


def parse_dimacs(path: Path) -> CNFFormula:
    num_vars: int | None = None
    expected_clauses: int | None = None
    clauses: list[tuple[int, ...]] = []
    current_clause: list[int] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) != 4 or parts[1].lower() != "cnf":
                    raise ValueError(f"Invalid DIMACS header in {path}: {line}")
                num_vars = int(parts[2])
                expected_clauses = int(parts[3])
                continue

            for token in line.split():
                literal = int(token)
                if literal == 0:
                    clauses.append(tuple(current_clause))
                    current_clause = []
                else:
                    current_clause.append(literal)

    if current_clause:
        raise ValueError(f"Unterminated clause in {path}")
    if num_vars is None or expected_clauses is None:
        raise ValueError(f"Missing DIMACS header in {path}")
    if expected_clauses != len(clauses):
        raise ValueError(
            f"Clause count mismatch in {path}: header={expected_clauses}, parsed={len(clauses)}"
        )

    for clause in clauses:
        if not clause:
            raise ValueError(f"Empty clause encountered in {path}")
        for literal in clause:
            if abs(literal) < 1 or abs(literal) > num_vars:
                raise ValueError(f"Literal {literal} outside variable range in {path}")

    return CNFFormula(path=path, num_vars=num_vars, clauses=tuple(clauses))


def assignment_satisfies(
    clauses: tuple[tuple[int, ...], ...], assignment: dict[int, bool]
) -> bool:
    for clause in clauses:
        if not any(assignment.get(abs(literal)) == (literal > 0) for literal in clause):
            return False
    return True
