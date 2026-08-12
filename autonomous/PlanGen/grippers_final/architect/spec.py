"""Architect specification for the grippers ABR solver."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


Ball = str
Gripper = str
Robot = str
Room = str


@dataclass
class GrippersState:
    """Concrete state schema selected by the architect."""

    robots: List[Robot]
    rooms: List[Room]
    balls: List[Ball]
    grippers: List[Gripper]
    robot_at: Dict[Robot, Room]
    ball_at: Dict[Ball, Room]
    carrying: Dict[Gripper, Ball]
    free: Set[Gripper] = field(default_factory=set)


@dataclass
class GrippersGoals:
    """Goal predicates supported by the grippers planner."""

    robot_at: Dict[Robot, Room] = field(default_factory=dict)
    ball_at: Dict[Ball, Room] = field(default_factory=dict)
    carrying: Dict[Gripper, Ball] = field(default_factory=dict)
    free: Set[Gripper] = field(default_factory=set)
    ball_location_conflict: bool = False
    robot_location_conflict: bool = False
    gripper_carry_conflict: bool = False
    ball_carry_conflict: bool = False


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "GrippersState(robots, rooms, balls, grippers, robot_at, ball_at, carrying, free)"
    ),
    "goal_representation": (
        "GrippersGoals(robot_at, ball_at, carrying, free, and explicit conflict flags)"
    ),
    "algorithm": (
        "deterministic constructive planner for the fully connected grippers domain, "
        "with static consistency checks for mutually exclusive goal facts"
    ),
    "abstract_methods": [
        "parse_instance(context, inputs) -> (GrippersState, GrippersGoals)",
        "validate_goal_consistency(state, goals) -> (bool, reason)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> GrippersState",
        "goals_hold(state, goals) -> bool",
    ],
}
