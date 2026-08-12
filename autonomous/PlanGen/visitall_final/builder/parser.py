"""Natural-language parser for ACPBench visitall instances."""

import re
from typing import Dict, Set, Tuple

from architect.spec import Location, VisitAllGoals, VisitAllState


LOC_RE = r"loc-x\d+-y\d+"


def parse_instance(context: str, inputs: str) -> Tuple[VisitAllState, VisitAllGoals]:
    width, height = _parse_grid_size(context)
    unavailable = _parse_unavailable(context)
    locations = {
        _loc(x, y)
        for x in range(width)
        for y in range(height)
        if _loc(x, y) not in unavailable
    }
    current = _parse_current(context)
    visited = _parse_initial_visited(context)

    if current:
        locations.add(current)
        visited.add(current)

    connected = _build_grid_edges(width, height, locations)
    goals = parse_goals(inputs)
    return (
        VisitAllState(
            width=width,
            height=height,
            locations=locations,
            unavailable=unavailable,
            connected=connected,
            current=current,
            visited=visited,
        ),
        goals,
    )


def parse_goals(inputs: str) -> VisitAllGoals:
    goal_text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    facts = _split_goal_facts(goal_text)
    goals = VisitAllGoals(visited=set())

    for fact in facts:
        fact = fact.strip().rstrip(".?")
        locs = re.findall(LOC_RE, fact)
        if not locs:
            continue
        loc = locs[-1]
        low = fact.lower()
        if "robot" in low and (" at " in low or "in place" in low):
            if goals.at is not None and goals.at != loc:
                goals.at_conflict = True
            goals.at = loc
            continue
        if "visited" in low:
            goals.visited.add(loc)

    return goals


def _parse_grid_size(context: str) -> Tuple[int, int]:
    match = re.search(r"grid size is\s+(\d+)x(\d+)", context, re.I)
    if not match:
        raise ValueError("could not parse visitall grid size")
    return int(match.group(1)), int(match.group(2))


def _parse_unavailable(context: str) -> Set[Location]:
    if re.search(r"no unavailable cells", context, re.I):
        return set()
    segment = ""
    match = re.search(r"unavailable cells? (?:are|is)\s+(.*?)(?:\.|\n)", context, re.I | re.S)
    if match:
        segment = match.group(1)
    return set(re.findall(LOC_RE, segment))


def _parse_current(context: str) -> Location:
    match = re.search(r"Currently,\s+the robot is in place\s+(" + LOC_RE + r")", context, re.I)
    if not match:
        match = re.search(r"Currently,\s+the robot is at\s+(" + LOC_RE + r")", context, re.I)
    if not match:
        raise ValueError("could not parse current robot location")
    return match.group(1)


def _parse_initial_visited(context: str) -> Set[Location]:
    visited: Set[Location] = set()
    single = re.search(r"Place\s+(" + LOC_RE + r")\s+has been visited", context, re.I)
    if single:
        visited.add(single.group(1))

    listed = re.search(
        r"The following places have been visited:\s*(.*?)(?:\.|$)",
        context,
        re.I | re.S,
    )
    if listed:
        visited.update(re.findall(LOC_RE, listed.group(1)))
    return visited


def _build_grid_edges(width: int, height: int, locations: Set[Location]) -> Dict[Location, Set[Location]]:
    edges: Dict[Location, Set[Location]] = {loc: set() for loc in locations}
    for loc in locations:
        x, y = _xy(loc)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nxt = _loc(nx, ny)
            if 0 <= nx < width and 0 <= ny < height and nxt in locations:
                edges[loc].add(nxt)
    return edges


def _split_goal_facts(text: str):
    normalized = re.sub(r"\s+", " ", text)
    return re.split(r", and | and |, ", normalized)


def _loc(x: int, y: int) -> Location:
    return f"loc-x{x}-y{y}"


def _xy(loc: Location) -> Tuple[int, int]:
    match = re.fullmatch(r"loc-x(\d+)-y(\d+)", loc)
    if not match:
        raise ValueError(f"invalid location name: {loc}")
    return int(match.group(1)), int(match.group(2))
