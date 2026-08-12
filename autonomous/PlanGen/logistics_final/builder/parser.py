"""Natural-language parser for ACPBench logistics instances."""

import re
from typing import Dict, List, Tuple

from architect.spec import LogisticsGoals, LogisticsState


CITY_RE = r"c\d+"
LOC_RE = r"l\d+-\d+"
PACKAGE_RE = r"p\d+"
TRUCK_RE = r"t\d+"
AIRPLANE_RE = r"a\d+"
OBJECT_RE = rf"(?:{PACKAGE_RE}|{TRUCK_RE}|{AIRPLANE_RE})"


def parse_instance(context: str, inputs: str) -> Tuple[LogisticsState, LogisticsGoals]:
    location_city = _parse_location_cities(context)
    locations = sorted(location_city, key=_location_key)
    cities = sorted(set(location_city.values()), key=_object_key)
    city_airport = _infer_city_airports(location_city)

    text = f"{context} {inputs}"
    packages = sorted(set(re.findall(PACKAGE_RE, text)), key=_object_key)
    trucks = sorted(set(re.findall(TRUCK_RE, text)), key=_object_key)
    airplanes = sorted(set(re.findall(AIRPLANE_RE, text)), key=_object_key)
    at, in_vehicle = _parse_initial_positions(context)
    goals = parse_goals(inputs)

    return (
        LogisticsState(
            cities=cities,
            locations=locations,
            packages=packages,
            trucks=trucks,
            airplanes=airplanes,
            location_city=location_city,
            city_airport=city_airport,
            at=at,
            in_vehicle=in_vehicle,
        ),
        goals,
    )


def parse_goals(inputs: str) -> LogisticsGoals:
    text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    facts = _split_goal_facts(text)
    goals = LogisticsGoals()
    seen_at: Dict[str, List[str]] = {}
    seen_in: Dict[str, List[str]] = {}

    for raw_fact in facts:
        fact = raw_fact.strip().rstrip(".?")
        at_match = re.fullmatch(rf"({OBJECT_RE}) is at ({LOC_RE})", fact, re.I)
        if at_match:
            obj, loc = at_match.groups()
            seen_at.setdefault(obj, []).append(loc)
            goals.at[obj] = loc
            continue

        in_match = re.fullmatch(rf"({PACKAGE_RE}) is in ({TRUCK_RE}|{AIRPLANE_RE})", fact, re.I)
        if in_match:
            package, vehicle = in_match.groups()
            seen_in.setdefault(package, []).append(vehicle)
            goals.in_vehicle[package] = vehicle

    goals.object_location_conflict = any(len(set(values)) > 1 for values in seen_at.values())
    goals.object_container_conflict = any(len(set(values)) > 1 for values in seen_in.values())
    goals.object_mixed_conflict = bool(set(seen_at) & set(seen_in))
    return goals


def _parse_location_cities(context: str) -> Dict[str, str]:
    section = context.split("The locations are in cities as follows:", 1)[-1]
    section = section.split("Currently,", 1)[0]
    location_city: Dict[str, str] = {}
    for clause in section.split(";"):
        city_match = re.search(rf"are in ({CITY_RE})", clause)
        if not city_match:
            continue
        city = city_match.group(1)
        for loc in re.findall(LOC_RE, clause):
            location_city[loc] = city
    return location_city


def _infer_city_airports(location_city: Dict[str, str]) -> Dict[str, str]:
    airports: Dict[str, str] = {}
    for loc, city in location_city.items():
        suffix = loc.split("-", 1)[1]
        if suffix == "0":
            airports[city] = loc
    return airports


def _parse_initial_positions(context: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    current = context.split("Currently,", 1)[-1].strip().rstrip(".")
    at: Dict[str, str] = {}
    in_vehicle: Dict[str, str] = {}

    pattern = rf"(.+?)\s+(?:are|is)\s+(at|in)\s+({LOC_RE}|{TRUCK_RE}|{AIRPLANE_RE})(?:,|\.|$)"
    for objects_text, relation, target in re.findall(pattern, current, re.I):
        for obj in re.findall(OBJECT_RE, objects_text):
            if relation.lower() == "at":
                at[obj] = target
            else:
                in_vehicle[obj] = target
    return at, in_vehicle


def _split_goal_facts(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text)
    return re.split(r", and | and |, ", normalized)


def _object_key(name: str):
    match = re.match(r"([A-Za-z_]+)(\d+)$", name)
    if not match:
        return (name, -1)
    return (match.group(1), int(match.group(2)))


def _location_key(name: str):
    match = re.match(r"l(\d+)-(\d+)$", name)
    if not match:
        return (name, -1, -1)
    return ("l", int(match.group(1)), int(match.group(2)))
