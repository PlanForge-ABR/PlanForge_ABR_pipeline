from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ArchitectSpec:
    """Executable ABR design for NaturalPlan trip-planning instances."""

    representation: str = (
        "A trip is an ordered path over cities. Each city has an inclusive "
        "visit interval [start_day, end_day]. Consecutive cities overlap on "
        "the flight day, so next_start == current_end."
    )
    strategy: str = (
        "Parse cities, durations, total days, flights, and fixed timed events; "
        "then run deterministic constrained DFS over feasible flight paths."
    )


@dataclass(frozen=True)
class EventConstraint:
    city: str
    start: int
    end: int


@dataclass(frozen=True)
class Problem:
    key: str
    num_cities: int
    cities: Tuple[str, ...]
    durations: Dict[str, int]
    total_days: int
    directed_edges: Set[Tuple[str, str]]
    events: Tuple[EventConstraint, ...]


@dataclass(frozen=True)
class Visit:
    city: str
    start: int
    end: int
    duration: int


@dataclass(frozen=True)
class RunnerResult:
    status: str
    plan: Optional[str]
    visits: Tuple[Visit, ...] = ()
    reason: Optional[str] = None


def parse_problem(key: str, instance: dict) -> Problem:
    cities = tuple(part.strip() for part in instance["cities"].split("**"))
    durations_raw = [int(part) for part in instance["durations"].split("**")]
    durations = dict(zip(cities, durations_raw))
    prompt = instance["prompt_0shot"]
    total_match = re.search(r"for (\d+) days in total", prompt)
    if not total_match:
        raise ValueError(f"{key}: could not parse total days")

    return Problem(
        key=key,
        num_cities=int(instance["num_cities"]),
        cities=cities,
        durations=durations,
        total_days=int(total_match.group(1)),
        directed_edges=parse_flights(prompt, cities),
        events=tuple(parse_events(prompt, cities)),
    )


def parse_flights(prompt: str, cities: Sequence[str]) -> Set[Tuple[str, str]]:
    section_match = re.search(
        r"Here are the cities that have direct flights:\s*(.*?)\s*Find a trip",
        prompt,
        flags=re.S,
    )
    if not section_match:
        return set()

    city_set = set(cities)
    edges: Set[Tuple[str, str]] = set()
    for segment in section_match.group(1).replace("\n", " ").split(","):
        text = segment.strip().strip(".")
        if not text:
            continue
        directed = re.fullmatch(r"from (.+?) to (.+)", text)
        undirected = re.fullmatch(r"(.+?) and (.+)", text)
        if directed:
            src, dst = directed.group(1).strip(), directed.group(2).strip()
            if src in city_set and dst in city_set:
                edges.add((src, dst))
        elif undirected:
            left, right = undirected.group(1).strip(), undirected.group(2).strip()
            if left in city_set and right in city_set:
                edges.add((left, right))
                edges.add((right, left))
    return edges


def parse_events(prompt: str, cities: Sequence[str]) -> List[EventConstraint]:
    before_flights = prompt.split("Here are the cities that have direct flights:")[0]
    events: List[EventConstraint] = []
    city_pattern = "|".join(re.escape(city) for city in sorted(cities, key=len, reverse=True))

    patterns = [
        re.compile(
            rf"(?:in|at) (?P<city>{city_pattern}) between day (?P<start>\d+) and day (?P<end>\d+)",
            flags=re.I,
        ),
        re.compile(
            rf"between day (?P<start>\d+) and day (?P<end>\d+).*?(?:in|at) (?P<city>{city_pattern})",
            flags=re.I,
        ),
        re.compile(
            rf"During day (?P<start>\d+) and day (?P<end>\d+).*?(?:in|at) (?P<city>{city_pattern})",
            flags=re.I,
        ),
        re.compile(
            rf"From day (?P<start>\d+) to day (?P<end>\d+).*?(?:in|at) (?P<city>{city_pattern})",
            flags=re.I,
        ),
    ]

    seen = set()
    sentences = [part.strip() for part in re.split(r"(?<=\.)\s+", before_flights) if part.strip()]
    for sentence in sentences:
        for pattern in patterns:
            for match in pattern.finditer(sentence):
                event = EventConstraint(
                    city=match.group("city"),
                    start=int(match.group("start")),
                    end=int(match.group("end")),
                )
                marker = (event.city, event.start, event.end)
                if marker not in seen:
                    seen.add(marker)
                    events.append(event)
    return events


def solve_problem(problem: Problem) -> RunnerResult:
    expected_total = sum(problem.durations.values()) - (problem.num_cities - 1)
    if expected_total != problem.total_days:
        return RunnerResult("FAILURE", None, reason="duration total is inconsistent")

    adjacency: Dict[str, List[str]] = {city: [] for city in problem.cities}
    for src, dst in sorted(problem.directed_edges):
        adjacency.setdefault(src, []).append(dst)

    best = search_path(problem, adjacency)
    if best is None:
        return RunnerResult("FAILURE", None, reason="no valid flight path satisfies constraints")
    if not verify_visits(problem, best):
        return RunnerResult("FAILURE", None, visits=tuple(best), reason="internal verification failed")
    return RunnerResult("SUCCESS", format_plan(problem, best), visits=tuple(best))


def search_path(problem: Problem, adjacency: Dict[str, List[str]]) -> Optional[List[Visit]]:
    event_by_city: Dict[str, List[EventConstraint]] = {city: [] for city in problem.cities}
    for event in problem.events:
        event_by_city[event.city].append(event)

    def compatible_interval(city: str, start: int, end: int) -> bool:
        return all(start <= event.start and end >= event.end for event in event_by_city[city])

    def city_can_still_fit(city: str, earliest_start: int) -> bool:
        duration = problem.durations[city]
        latest_possible_start = problem.total_days - duration + 1
        if earliest_start > latest_possible_start:
            return False
        for event in event_by_city[city]:
            if earliest_start > event.start:
                return False
            if latest_possible_start + duration - 1 < event.end:
                return False
        return True

    def candidate_score(city: str, next_start: int) -> Tuple[int, int, int]:
        duration = problem.durations[city]
        end = next_start + duration - 1
        exact_event = any(next_start <= event.start and end >= event.end for event in event_by_city[city])
        next_event_start = min((event.start for event in event_by_city[city]), default=999)
        original_index = problem.cities.index(city)
        return (0 if exact_event else 1, abs(next_event_start - next_start), original_index)

    def dfs(city: str, start: int, visited: Set[str], visits: List[Visit]) -> Optional[List[Visit]]:
        duration = problem.durations[city]
        end = start + duration - 1
        if end > problem.total_days or not compatible_interval(city, start, end):
            return None

        visit = Visit(city=city, start=start, end=end, duration=duration)
        next_visits = visits + [visit]
        next_visited = visited | {city}

        if len(next_visited) == problem.num_cities:
            if end == problem.total_days:
                return next_visits
            return None

        next_start = end
        remaining = set(problem.cities) - next_visited
        if any(not city_can_still_fit(other, next_start) for other in remaining):
            return None

        candidates = [dst for dst in adjacency.get(city, []) if dst in remaining]
        candidates.sort(key=lambda dst: candidate_score(dst, next_start))
        for dst in candidates:
            found = dfs(dst, next_start, next_visited, next_visits)
            if found is not None:
                return found
        return None

    starts = sorted(problem.cities, key=lambda city: candidate_score(city, 1))
    for city in starts:
        found = dfs(city, 1, set(), [])
        if found is not None:
            return found

    # Dense high-city cases may have many valid edges and sparse events. The DFS
    # above is usually enough, but this deterministic fallback helps avoid a bad
    # early branch when the first ordering heuristic is underconstrained.
    if problem.num_cities <= 8:
        for order in permutations(problem.cities):
            visits: List[Visit] = []
            start = 1
            valid = True
            for index, city in enumerate(order):
                end = start + problem.durations[city] - 1
                if not compatible_interval(city, start, end):
                    valid = False
                    break
                visits.append(Visit(city, start, end, problem.durations[city]))
                if index < len(order) - 1 and (city, order[index + 1]) not in problem.directed_edges:
                    valid = False
                    break
                start = end
            if valid and visits[-1].end == problem.total_days:
                return visits
    return None


def verify_visits(problem: Problem, visits: Sequence[Visit]) -> bool:
    if len(visits) != problem.num_cities:
        return False
    if {visit.city for visit in visits} != set(problem.cities):
        return False
    if visits[0].start != 1 or visits[-1].end != problem.total_days:
        return False
    for visit in visits:
        if visit.duration != problem.durations[visit.city]:
            return False
        if visit.end - visit.start + 1 != visit.duration:
            return False
    for left, right in zip(visits, visits[1:]):
        if left.end != right.start:
            return False
        if (left.city, right.city) not in problem.directed_edges:
            return False
    for event in problem.events:
        visit = next((item for item in visits if item.city == event.city), None)
        if visit is None or not (visit.start <= event.start and visit.end >= event.end):
            return False
    return True


def format_plan(problem: Problem, visits: Sequence[Visit]) -> str:
    lines = [
        f"Here is the trip plan for visiting the {problem.num_cities} European cities for {problem.total_days} days:",
        "",
    ]
    for index, visit in enumerate(visits):
        if index == 0:
            visit_text = f"Arriving in {visit.city} and visit {visit.city}"
        else:
            visit_text = f"Visit {visit.city}"
        lines.append(f"**Day {visit.start}-{visit.end}:** {visit_text} for {visit.duration} days.")
        if index + 1 < len(visits):
            next_city = visits[index + 1].city
            lines.append(f"**Day {visit.end}:** Fly from {visit.city} to {next_city}.")
    return "\n".join(lines)


def architect_spec() -> ArchitectSpec:
    return ArchitectSpec()


def builder_run(key: str, instance: dict) -> RunnerResult:
    problem = parse_problem(key, instance)
    return solve_problem(problem)


def runner_execute(key: str, instance: dict) -> dict:
    result = builder_run(key, instance)
    return {"status": result.status, "plan": result.plan}
