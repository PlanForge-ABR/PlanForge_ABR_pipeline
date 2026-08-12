"""Architect specification for the alfworld ABR solver."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class AWState:
    """Symbolic household state selected by the architect."""

    agent_location: str
    locations: Set[str]
    objects_by_type: Dict[str, List[str]]
    receptacles_by_type: Dict[str, List[str]]
    object_locations: Dict[str, str]
    receptacle_locations: Dict[str, str]
    object_receptacle: Dict[str, str]
    open_receptacles: Set[str]
    closed_receptacles: Set[str]
    toggled_objects: Set[str]
    validated: Set[str]
    holding: Optional[str] = None


@dataclass
class ValidationGoal:
    """A high-level ALFWorld task validation requested by the instance."""

    object_type: str
    receptacle_type: Optional[str] = None
    property_name: Optional[str] = None
    count: int = 1
    tool_type: Optional[str] = None

    @property
    def key(self) -> str:
        parts = [str(self.count), self.object_type, self.property_name or "", self.receptacle_type or "", self.tool_type or ""]
        return "|".join(parts)


@dataclass
class DirectGoal:
    """Simple ground fact goal."""

    kind: str
    subject: str
    value: Optional[str] = None


@dataclass
class AWGoals:
    validations: List[ValidationGoal] = field(default_factory=list)
    direct: List[DirectGoal] = field(default_factory=list)


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": (
        "AWState with agent location, typed objects/receptacles, object containment, "
        "open/closed receptacles, toggled objects, existing validations, and optional hand content"
    ),
    "goal_representation": "AWGoals containing direct fact goals and ALFWorld validation task goals",
    "algorithm": "deterministic symbolic task-template planner over parsed household facts",
    "abstract_methods": [
        "parse_instance(context, inputs) -> (AWState, AWGoals)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> AWState",
        "goals_hold(state, goals) -> bool",
    ],
}
