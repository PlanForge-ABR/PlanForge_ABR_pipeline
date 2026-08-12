"""State schema and algorithm choice for the rovers domain.

The architect selects a deterministic goal-directed planner over the STRIPS
state encoded in each benchmark instance. Navigation is solved with BFS on the
per-rover traversability graph; resource-producing goals are achieved by
constructing the required prerequisite facts and then executing the matching
domain action.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


Fact = Tuple[str, ...]


@dataclass
class RoverProblem:
    objects: Dict[str, List[str]]
    init_facts: Set[Fact]
    goal_facts: List[Fact]


@dataclass
class RoverState:
    rovers: List[str]
    waypoints: List[str]
    landers: List[str]
    cameras: List[str]
    objectives: List[str]
    modes: List[str]
    stores: List[str]
    at: Dict[str, str]
    lander_at: Dict[str, str]
    empty: Set[str]
    full: Set[str]
    calibrated: Set[Tuple[str, str]]
    have_rock: Set[Tuple[str, str]]
    have_soil: Set[Tuple[str, str]]
    have_image: Set[Tuple[str, str, str]]
    communicated_rock: Set[str]
    communicated_soil: Set[str]
    communicated_image: Set[Tuple[str, str]]
    rock_samples: Set[str]
    soil_samples: Set[str]
    static: Set[Fact]

    def copy(self) -> "RoverState":
        return RoverState(
            rovers=list(self.rovers),
            waypoints=list(self.waypoints),
            landers=list(self.landers),
            cameras=list(self.cameras),
            objectives=list(self.objectives),
            modes=list(self.modes),
            stores=list(self.stores),
            at=dict(self.at),
            lander_at=dict(self.lander_at),
            empty=set(self.empty),
            full=set(self.full),
            calibrated=set(self.calibrated),
            have_rock=set(self.have_rock),
            have_soil=set(self.have_soil),
            have_image=set(self.have_image),
            communicated_rock=set(self.communicated_rock),
            communicated_soil=set(self.communicated_soil),
            communicated_image=set(self.communicated_image),
            rock_samples=set(self.rock_samples),
            soil_samples=set(self.soil_samples),
            static=set(self.static),
        )


@dataclass
class PlanResult:
    exists: bool
    plan: List[str] = field(default_factory=list)
    reason: str = ""
    failed_goal: Optional[Fact] = None
