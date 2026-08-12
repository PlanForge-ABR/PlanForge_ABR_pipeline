"""Architect specification for the floortile ABR solver."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


Fact = Tuple[str, ...]


@dataclass
class FloorTileState:
    """State schema selected for the floor-tile domain."""

    colors: List[str]
    robots: List[str]
    tiles: List[str]
    move_edges: Dict[str, Dict[str, str]]
    robot_at: Dict[str, str]
    robot_has: Dict[str, str]
    clear: Set[str] = field(default_factory=set)
    painted: Dict[str, str] = field(default_factory=dict)


@dataclass
class FloorTileGoals:
    """Goal facts supported by the solver."""

    facts: List[Fact]


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "FloorTileState(colors, robots, tiles, directed movement edges, robot locations, "
        "robot colors, clear tiles, and immutable painted tile colors)"
    ),
    "goal_representation": "FloorTileGoals as parsed PDDL facts",
    "algorithm": "mutex validation followed by informed forward state-space search",
    "abstract_methods": [
        "parse_instance(item) -> (FloorTileState, FloorTileGoals)",
        "validate_goal_consistency(state, goals) -> (bool, reason)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> FloorTileState",
        "goals_hold(state, goals) -> bool",
    ],
}
