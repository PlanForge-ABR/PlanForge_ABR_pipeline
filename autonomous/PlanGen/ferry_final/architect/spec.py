"""Architect specification for the ferry ABR solver."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


Car = str
Location = str


@dataclass
class FerryState:
    """Concrete state schema selected by the architect."""

    cars: List[Car]
    locations: List[Location]
    car_at: Dict[Car, Location]
    ferry_at: Location
    onboard: Optional[Car] = None


@dataclass
class FerryGoals:
    """Goal predicates supported by the ferry planner."""

    car_at: Dict[Car, Location] = field(default_factory=dict)
    ferry_at: Optional[Location] = None
    onboard: Optional[Car] = None
    empty: bool = False
    car_location_conflict: bool = False
    ferry_location_conflict: bool = False
    onboard_conflict: bool = False


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "FerryState(cars, locations, car_at map for cars ashore, ferry_at, optional onboard car)"
    ),
    "goal_representation": (
        "FerryGoals(car_at map, optional ferry_at, optional onboard car, empty flag, conflict flags)"
    ),
    "algorithm": "deterministic constructive planner over fully connected ferry locations",
    "abstract_methods": [
        "parse_instance(context, inputs) -> (FerryState, FerryGoals)",
        "validate_goal_consistency(state, goals) -> (bool, reason)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> FerryState",
        "goals_hold(state, goals) -> bool",
    ],
}
