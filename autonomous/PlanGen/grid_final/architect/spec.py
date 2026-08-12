"""Architect specification for the grid ABR solver."""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple


Location = str
Key = str
Shape = str
GoalAtom = Tuple[str, str, Optional[str]]


@dataclass(frozen=True)
class GridState:
    """Concrete state schema selected by the architect."""

    rows: int
    cols: int
    robot: Location
    holding: Optional[Key]
    key_at: Tuple[Tuple[Key, Location], ...]
    key_shape: Tuple[Tuple[Key, Shape], ...]
    lock_shape: Tuple[Tuple[Location, Shape], ...]
    locked: FrozenSet[Location]


@dataclass(frozen=True)
class GridGoals:
    """Conjunctive goal facts supported by the planner."""

    atoms: FrozenSet[GoalAtom]


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "GridState(rows, cols, robot, optional held key, key locations, key shapes, lock shapes, locked cells)"
    ),
    "goal_representation": (
        "GridGoals as normalized atoms: robot-at, holding, key-at, arm-empty, open, locked"
    ),
    "algorithm": "A* graph search over deterministic STRIPS transitions with monotonic unlock effects",
    "abstract_methods": [
        "parse_instance(context, inputs) -> (GridState, GridGoals)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> GridState",
        "goals_hold(state, goals) -> bool",
    ],
}
