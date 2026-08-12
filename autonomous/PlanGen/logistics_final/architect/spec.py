"""Architect specification for the logistics ABR solver."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


ObjectName = str
Location = str
City = str


@dataclass
class LogisticsState:
    """Concrete state schema selected by the architect."""

    cities: List[City]
    locations: List[Location]
    packages: List[ObjectName]
    trucks: List[ObjectName]
    airplanes: List[ObjectName]
    location_city: Dict[Location, City]
    city_airport: Dict[City, Location]
    at: Dict[ObjectName, Location] = field(default_factory=dict)
    in_vehicle: Dict[ObjectName, ObjectName] = field(default_factory=dict)


@dataclass
class LogisticsGoals:
    """Goal predicates supported by the logistics planner."""

    at: Dict[ObjectName, Location] = field(default_factory=dict)
    in_vehicle: Dict[ObjectName, ObjectName] = field(default_factory=dict)
    object_location_conflict: bool = False
    object_container_conflict: bool = False
    object_mixed_conflict: bool = False


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "LogisticsState(cities, locations, packages, trucks, airplanes, location_city, "
        "city_airport, at, in_vehicle)"
    ),
    "goal_representation": (
        "LogisticsGoals(at, in_vehicle, and explicit contradiction flags for objects "
        "required to be in two places or both at and inside a vehicle)"
    ),
    "algorithm": (
        "deterministic constructive planner over the standard logistics topology: "
        "trucks drive only inside one city, airplanes fly only between city airport "
        "locations, and packages are transferred through airports for inter-city travel"
    ),
    "abstract_methods": [
        "parse_instance(context, inputs) -> (LogisticsState, LogisticsGoals)",
        "validate_goal_consistency(state, goals) -> (bool, reason)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> LogisticsState",
        "goals_hold(state, goals) -> bool",
    ],
}
