"""Architect specification for the depot ABR solver."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


Fact = Tuple[str, ...]


@dataclass
class DepotState:
    crates: List[str]
    pallets: List[str]
    hoists: List[str]
    trucks: List[str]
    places: List[str]
    at: Dict[str, str] = field(default_factory=dict)
    available: Set[str] = field(default_factory=set)
    clear: Set[str] = field(default_factory=set)
    on: Dict[str, str] = field(default_factory=dict)
    in_truck: Dict[str, str] = field(default_factory=dict)
    lifting: Dict[str, str] = field(default_factory=dict)


@dataclass
class DepotGoals:
    facts: List[Fact] = field(default_factory=list)


@dataclass
class SolveResult:
    exists: bool
    plan: List[str] = field(default_factory=list)
    reason: str = ""
