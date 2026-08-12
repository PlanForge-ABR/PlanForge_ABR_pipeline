from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Tuple

from architect.state_schema import EventConstraint, TripProblem


TOTAL_RE = re.compile(r"for (\d+) days in total", re.IGNORECASE)
EVENT_RE = re.compile(
    r"(?P<sentence>[^.\n]*(?:between day (?P<start1>\d+) and day (?P<end1>\d+)|"
    r"During day (?P<start2>\d+) and day (?P<end2>\d+))[^.\n]*\.)",
    re.IGNORECASE,
)
DIRECTED_RE = re.compile(r"^from\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE)


def parse_trip_example(example_id: str, raw: Dict[str, Any]) -> TripProblem:
    cities = tuple(part.strip() for part in raw["cities"].split("**") if part.strip())
    durations_raw = [int(part.strip()) for part in raw["durations"].split("**") if part.strip()]
    if len(cities) != len(durations_raw):
        raise ValueError(f"{example_id}: cities and durations length mismatch.")

    num_cities = int(raw["num_cities"])
    if num_cities != len(cities):
        raise ValueError(f"{example_id}: num_cities does not match city list.")

    prompt = raw["prompt_0shot"]
    total_match = TOTAL_RE.search(prompt)
    if not total_match:
        raise ValueError(f"{example_id}: could not parse total days.")

    problem = TripProblem(
        example_id=example_id,
        num_cities=num_cities,
        cities=cities,
        durations=dict(zip(cities, durations_raw)),
        total_days=int(total_match.group(1)),
        flights=_parse_flights(prompt, cities),
        constraints=tuple(_parse_event_constraints(prompt, cities)),
    )
    _validate_problem(problem)
    return problem


def _parse_flights(prompt: str, cities: Tuple[str, ...]) -> FrozenSet[Tuple[str, str]]:
    marker = "Here are the cities that have direct flights:"
    if marker not in prompt:
        raise ValueError("Missing direct-flight section.")
    section = prompt.split(marker, 1)[1].split("\n\nFind", 1)[0].strip().rstrip(".")
    edges: set[Tuple[str, str]] = set()
    city_set = set(cities)

    for part in re.split(r",\s*", section):
        part = part.strip().rstrip(".")
        if not part:
            continue
        directed = DIRECTED_RE.match(part)
        if directed:
            src, dst = directed.group(1).strip(), directed.group(2).strip()
            _ensure_city(src, city_set)
            _ensure_city(dst, city_set)
            edges.add((src, dst))
            continue
        if " and " in part:
            left, right = [piece.strip() for piece in part.split(" and ", 1)]
            _ensure_city(left, city_set)
            _ensure_city(right, city_set)
            edges.add((left, right))
            edges.add((right, left))
            continue
        raise ValueError(f"Unrecognized flight fragment: {part}")

    return frozenset(edges)


def _parse_event_constraints(prompt: str, cities: Tuple[str, ...]) -> List[EventConstraint]:
    constraints: List[EventConstraint] = []
    for match in EVENT_RE.finditer(prompt):
        sentence = match.group("sentence")
        city = _city_in_sentence(sentence, cities)
        if city is None:
            continue
        start = int(match.group("start1") or match.group("start2"))
        end = int(match.group("end1") or match.group("end2"))
        constraints.append(EventConstraint(city=city, start_day=start, end_day=end))
    return constraints


def _city_in_sentence(sentence: str, cities: Tuple[str, ...]) -> str | None:
    matches = [city for city in cities if re.search(rf"\b{re.escape(city)}\b", sentence)]
    if not matches:
        return None
    return max(matches, key=len)


def _ensure_city(city: str, city_set: set[str]) -> None:
    if city not in city_set:
        raise ValueError(f"Unknown city in parsed data: {city}")


def _validate_problem(problem: TripProblem) -> None:
    expected_total = sum(problem.durations.values()) - (problem.num_cities - 1)
    if expected_total != problem.total_days:
        raise ValueError(
            f"{problem.example_id}: duration total {expected_total} != stated total {problem.total_days}."
        )
    for constraint in problem.constraints:
        if constraint.city not in problem.durations:
            raise ValueError(f"{problem.example_id}: event city is not in city list.")
        if not (1 <= constraint.start_day <= constraint.end_day <= problem.total_days):
            raise ValueError(f"{problem.example_id}: event window is outside trip days.")

