from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


Minute = int
Interval = Tuple[Minute, Minute]


@dataclass(frozen=True)
class CalendarProblem:
    people: Tuple[str, ...]
    day: str
    days: Tuple[str, ...]
    work_start: Minute
    work_end: Minute
    duration: Minute
    busy: Dict[str, Dict[str, Tuple[Interval, ...]]]
    preferences: Tuple["Preference", ...] = ()


@dataclass(frozen=True)
class Preference:
    person: str
    day: str
    direction: str
    minute: Optional[Minute] = None
    hard: bool = False


@dataclass(frozen=True)
class CalendarState:
    assigned: bool = False
    day: Optional[str] = None
    start: Optional[Minute] = None
    end: Optional[Minute] = None
    checked_slots: Tuple[Minute, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CalendarAction:
    day: str
    start: Minute
    end: Minute
