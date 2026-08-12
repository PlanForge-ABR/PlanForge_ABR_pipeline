from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class EventConstraint:
    city: str
    start_day: int
    end_day: int


@dataclass(frozen=True)
class Segment:
    city: str
    start_day: int
    end_day: int
    duration: int


@dataclass(frozen=True)
class TripProblem:
    example_id: str
    num_cities: int
    cities: Tuple[str, ...]
    durations: Dict[str, int]
    total_days: int
    flights: FrozenSet[Tuple[str, str]]
    constraints: Tuple[EventConstraint, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TripState:
    current_city: Optional[str]
    visited_cities: Tuple[str, ...]
    remaining_cities: FrozenSet[str]
    total_days_used: int
    segments: Tuple[Segment, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SearchNode:
    state: TripState
    actions: Tuple[str, ...] = field(default_factory=tuple)

