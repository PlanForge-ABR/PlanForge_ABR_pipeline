"""Domain specification selected by the Architect for satellite planning.

The state is a compact STRIPS fact model and the runner uses constructive
planning over the monotonic/non-monotonic resources in the satellite domain.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


Fact = Tuple[str, ...]


@dataclass
class SatelliteState:
    directions: Set[str] = field(default_factory=set)
    instruments: Set[str] = field(default_factory=set)
    modes: Set[str] = field(default_factory=set)
    satellites: Set[str] = field(default_factory=set)
    on_board: Dict[str, str] = field(default_factory=dict)
    supports: Dict[str, Set[str]] = field(default_factory=dict)
    calibration_targets: Dict[str, List[str]] = field(default_factory=dict)
    pointing: Dict[str, str] = field(default_factory=dict)
    power_avail: Set[str] = field(default_factory=set)
    power_on: Set[str] = field(default_factory=set)
    calibrated: Set[str] = field(default_factory=set)
    have_images: Set[Tuple[str, str]] = field(default_factory=set)


@dataclass
class PlanResult:
    exists: bool
    plan: List[str] = field(default_factory=list)
    reason: str = ""


ACTION_SCHEMAS = {
    "turn_to": "turn_to <satellite> <new_direction> <previous_direction>",
    "switch_on": "switch_on <instrument> <satellite>",
    "switch_off": "switch_off <instrument> <satellite>",
    "calibrate": "calibrate <satellite> <instrument> <target_direction>",
    "take_image": "take_image <satellite> <direction> <instrument> <mode>",
}


ABSTRACT_METHODS = [
    "parse_instance(item) -> tuple[SatelliteState, list[Fact]]",
    "construct_plan(state, goals) -> PlanResult",
    "simulate_plan(state, plan) -> SatelliteState",
    "goals_hold(state, goals) -> bool",
]
