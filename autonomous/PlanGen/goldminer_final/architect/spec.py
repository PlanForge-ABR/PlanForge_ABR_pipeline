"""Architect specification for the goldminer ABR solver."""

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple


Loc = str
GoalAtom = Tuple[str, Optional[Loc]]


@dataclass(frozen=True)
class GMState:
    """Concrete state schema selected by the architect."""

    rows: int
    cols: int
    robot: Loc
    holding: str
    bomb_at: Loc
    laser_at: Optional[Loc]
    clear: FrozenSet[Loc]
    soft: FrozenSet[Loc]
    hard: FrozenSet[Loc]
    gold: FrozenSet[Loc]


@dataclass(frozen=True)
class GMGoals:
    """Supported goal predicates parsed from the instance request."""

    atoms: FrozenSet[GoalAtom]


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "GMState(rows, cols, robot location, held item, bomb supply location, "
        "laser location, clear cells, soft rocks, hard rocks, gold cells)"
    ),
    "goal_representation": "GMGoals as normalized predicate/location atoms",
    "algorithm": "deterministic best-first graph search over executable domain actions",
    "abstract_methods": [
        "parse_instance(context, inputs) -> (GMState, GMGoals)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> GMState",
        "goals_hold(state, goals) -> bool",
    ],
}
