"""Architect specification for the hanoi ABR solver."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


Obj = str
Disk = str
OnFact = Tuple[Obj, Obj]


@dataclass
class HanoiState:
    """Concrete state schema selected by the architect."""

    disks: List[Disk]
    pegs: List[Obj]
    on: Dict[Disk, Obj]


@dataclass
class HanoiGoals:
    """Supported goal predicates parsed from the instance request."""

    on: Set[OnFact]
    clear: Set[Obj]


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": "HanoiState(disks ordered smallest to largest, pegs, on child_disk->support)",
    "goal_representation": "HanoiGoals(on direct child->support facts, clear objects)",
    "algorithm": "goal consistency check plus deterministic recursive/short-horizon constructive planning",
    "abstract_methods": [
        "parse_instance(context, inputs) -> (HanoiState, HanoiGoals)",
        "validate_goal_consistency(state, goals) -> (bool, reason)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> HanoiState",
        "goals_hold(state, goals) -> bool",
    ],
}
