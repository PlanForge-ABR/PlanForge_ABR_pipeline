"""Architect specification for the blocksworld ABR solver."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


Block = str
OnFact = Tuple[Block, Block]


@dataclass
class BWState:
    """Concrete state schema selected by the architect."""

    blocks: List[Block]
    on: Dict[Block, Block]
    ontable: Set[Block]
    holding: Optional[Block]


@dataclass
class BWGoals:
    """Supported goal predicates parsed from the instance request."""

    on: Set[OnFact]
    clear: Set[Block]
    ontable: Set[Block]
    holding: Optional[Block] = None
    holding_conflict: bool = False
    handempty: bool = False


@dataclass
class SolveResult:
    exists: bool
    plan: List[str]
    reason: str = ""


SPECIFICATION = {
    "state_representation": "BWState(blocks, on child->support, ontable set, optional holding block)",
    "goal_representation": "BWGoals(on pairs, clear blocks, ontable blocks, optional holding, handempty)",
    "algorithm": "deterministic constructive planner",
    "abstract_methods": [
        "parse_instance(context, inputs) -> (BWState, BWGoals)",
        "validate_goal_consistency(state, goals) -> (bool, reason)",
        "construct_plan(state, goals) -> SolveResult",
        "simulate_plan(state, plan) -> BWState",
        "goals_hold(state, goals) -> bool",
    ],
}
