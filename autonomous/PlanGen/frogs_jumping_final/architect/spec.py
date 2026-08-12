"""Architect specification for the frogs_jumping domain.

The selected representation is a one-dimensional board with one empty lilypad.
The selected algorithm is deterministic monotone search: first reject static
impossibilities, then advance by legal directional slides and jumps until the
requested partial board facts hold.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple


AtGoal = Tuple[str, int]


@dataclass(frozen=True)
class FrogState:
    """Current board occupancy for left-facing and right-facing frogs."""

    board: Tuple[str, ...]
    positions: Tuple[Tuple[str, int], ...]
    left_frogs: FrozenSet[str]
    right_frogs: FrozenSet[str]


@dataclass(frozen=True)
class FrogGoals:
    """Requested at/empty facts."""

    at: Tuple[AtGoal, ...]
    empty: Tuple[int, ...]


@dataclass(frozen=True)
class SolveResult:
    """Runner-facing result produced by the builder methods."""

    exists: bool
    plan: Tuple[str, ...]
    reason: str = ""


def position_map(state: FrogState) -> Dict[str, int]:
    return dict(state.positions)
