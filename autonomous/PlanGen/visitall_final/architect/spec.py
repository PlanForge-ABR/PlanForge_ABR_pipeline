"""Architect specification for the visitall ABR solver."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


Location = str


@dataclass
class VisitAllState:
    """Concrete state schema selected by the architect."""

    width: int
    height: int
    locations: Set[Location]
    unavailable: Set[Location]
    connected: Dict[Location, Set[Location]]
    current: Location
    visited: Set[Location]


@dataclass
class VisitAllGoals:
    """Supported goal predicates parsed from the instance request."""

    visited: Set[Location]
    at: Optional[Location] = None
    at_conflict: bool = False


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "VisitAllState(width, height, available locations, blocked locations, "
        "adjacency map, current robot location, monotonic visited set)"
    ),
    "goal_representation": "VisitAllGoals(visited locations, optional final at location, at conflict flag)",
    "algorithm": "deterministic graph route construction using BFS shortest paths between required waypoints",
    "abstract_methods": [
        "parse_instance(context, inputs) -> (VisitAllState, VisitAllGoals)",
        "validate_goal_consistency(state, goals) -> (bool, reason)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> VisitAllState",
        "goals_hold(state, goals) -> bool",
    ],
}
