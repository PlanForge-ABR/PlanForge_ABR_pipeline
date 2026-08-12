"""Strict state and problem schemas for meeting planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FriendConstraint:
    index: int
    name: str
    location: str
    window_start: int
    window_end: int
    min_duration: int


@dataclass(frozen=True)
class MeetingAction:
    friend_index: int
    friend_name: str
    origin: str
    destination: str
    depart_time: int
    travel_minutes: int
    arrival_time: int
    wait_until: int | None
    meeting_start: int
    meeting_end: int
    duration: int


@dataclass(frozen=True)
class MeetingState:
    current_location: str
    current_time: int
    visited_people: Tuple[int, ...]
    actions: Tuple[MeetingAction, ...]


@dataclass(frozen=True)
class MeetingProblem:
    example_id: str
    start_location: str
    start_time: int
    friends: Tuple[FriendConstraint, ...]
    dist_matrix: dict[str, dict[str, int]]

