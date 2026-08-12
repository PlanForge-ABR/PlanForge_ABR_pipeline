"""Natural-language parser for ACPBench ferry instances."""

import re
from typing import Dict, List, Tuple

from architect.spec import FerryGoals, FerryState


CAR_RE = r"c\d+"
LOC_RE = r"l\d+"


def parse_instance(context: str, inputs: str) -> Tuple[FerryState, FerryGoals]:
    locations = [f"l{i}" for i in range(_parse_count(context, "locations"))]
    cars = [f"c{i}" for i in range(_parse_count(context, "cars"))]

    ferry_at = _parse_ferry_location(context)
    onboard = _parse_onboard_context(context)
    car_at = _parse_car_locations(context)

    for car in sorted(set(re.findall(CAR_RE, context + " " + inputs)), key=_object_key):
        if car not in cars:
            cars.append(car)
    for loc in sorted(set(re.findall(LOC_RE, context + " " + inputs)), key=_object_key):
        if loc not in locations:
            locations.append(loc)

    goals = parse_goals(inputs)
    return FerryState(cars=cars, locations=locations, car_at=car_at, ferry_at=ferry_at, onboard=onboard), goals


def _parse_count(context: str, noun: str) -> int:
    match = re.search(rf"There are (\d+) {noun}", context, re.I)
    return int(match.group(1)) if match else 0


def _parse_ferry_location(context: str) -> str:
    match = re.search(r"Currently, the ferry is at (l\d+)(?: location)?", context, re.I)
    if not match:
        raise ValueError("could not parse ferry location")
    return match.group(1)


def _parse_onboard_context(context: str):
    match = re.search(r"with the car (c\d+) on board", context, re.I)
    return match.group(1) if match else None


def _parse_car_locations(context: str) -> Dict[str, str]:
    if "The cars are at locations as follows:" not in context:
        return {}
    text = context.split("The cars are at locations as follows:", 1)[1].strip().rstrip(".")
    car_at: Dict[str, str] = {}
    for clause in re.split(r";\s*", text):
        loc_match = re.search(r"\bare at (l\d+)|\bis at (l\d+)", clause)
        if not loc_match:
            continue
        loc = loc_match.group(1) or loc_match.group(2)
        for car in re.findall(CAR_RE, clause):
            car_at[car] = loc
    return car_at


def parse_goals(inputs: str) -> FerryGoals:
    text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    facts = _split_goal_facts(text)
    goals = FerryGoals()

    seen_ferry_locations: List[str] = []
    seen_onboard: List[str] = []

    for fact in facts:
        fact = fact.strip().rstrip(".?")
        low = fact.lower()

        if "no cars on the ferry" in low or "ferry is empty" in low:
            goals.empty = True
            continue

        ferry_match = re.search(r"ferry is at (l\d+)(?: location)?", fact, re.I)
        if ferry_match:
            seen_ferry_locations.append(ferry_match.group(1))
            continue

        onboard_match = re.search(r"(?:car (c\d+) is on(?: board)? the ferry|ferry has car (c\d+) on board)", fact, re.I)
        if onboard_match:
            seen_onboard.append(onboard_match.group(1) or onboard_match.group(2))
            continue

        car_loc_match = re.search(r"car (c\d+) is at location (l\d+)", fact, re.I)
        if car_loc_match:
            car, loc = car_loc_match.groups()
            if car in goals.car_at and goals.car_at[car] != loc:
                goals.car_location_conflict = True
            goals.car_at[car] = loc

    distinct_ferry_locations = set(seen_ferry_locations)
    if len(distinct_ferry_locations) > 1:
        goals.ferry_location_conflict = True
    elif seen_ferry_locations:
        goals.ferry_at = seen_ferry_locations[-1]

    distinct_onboard = set(seen_onboard)
    if len(distinct_onboard) > 1:
        goals.onboard_conflict = True
    elif seen_onboard:
        goals.onboard = seen_onboard[-1]

    return goals


def _split_goal_facts(text: str):
    normalized = re.sub(r"\s+", " ", text)
    return re.split(r", and | and |, ", normalized)


def _object_key(name: str):
    match = re.match(r"([A-Za-z_]+)(\d+)$", name)
    if not match:
        return (name, -1)
    return (match.group(1), int(match.group(2)))
