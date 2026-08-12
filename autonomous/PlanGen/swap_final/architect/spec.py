"""Architect specification for the swap domain.

The selected representation is a finite bijection from agents to roles. The
selected algorithm is deterministic cycle repair by legal pairwise swaps.
"""

from dataclasses import dataclass
from typing import FrozenSet, Tuple


GoalAtom = Tuple[str, str]


@dataclass(frozen=True)
class SwapState:
    """Current one-to-one assignment of roles to agents."""

    assignments: Tuple[Tuple[str, str], ...]
    agents: FrozenSet[str]
    roles: FrozenSet[str]


@dataclass(frozen=True)
class SwapGoals:
    """Requested assignment facts."""

    atoms: Tuple[GoalAtom, ...]


@dataclass(frozen=True)
class SolveResult:
    """Runner-facing result produced by the builder methods."""

    exists: bool
    plan: Tuple[str, ...]
    reason: str = ""
