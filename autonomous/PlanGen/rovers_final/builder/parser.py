"""PDDL parser for ACPBench rovers instances."""

import re
from typing import Dict, List, Set, Tuple

from architect.spec import Fact, RoverProblem, RoverState


def parse_instance(item: dict) -> RoverProblem:
    problem = item["PDDL_problem"]
    objects = _parse_objects(problem)
    init_facts = set(_parse_section_facts(problem, ":init"))
    goal_facts = _parse_section_facts(problem, ":goal")
    return RoverProblem(objects=objects, init_facts=init_facts, goal_facts=goal_facts)


def make_state(problem: RoverProblem) -> RoverState:
    facts = problem.init_facts
    static_names = {
        "available",
        "calibration_target",
        "can_traverse",
        "channel_free",
        "equipped_for_imaging",
        "equipped_for_rock_analysis",
        "equipped_for_soil_analysis",
        "on_board",
        "store_of",
        "supports",
        "visible",
        "visible_from",
    }
    return RoverState(
        rovers=problem.objects.get("rover", []),
        waypoints=problem.objects.get("waypoint", []),
        landers=problem.objects.get("lander", []),
        cameras=problem.objects.get("camera", []),
        objectives=problem.objects.get("objective", []),
        modes=problem.objects.get("mode", []),
        stores=problem.objects.get("store", []),
        at={f[1]: f[2] for f in facts if f[0] == "at"},
        lander_at={f[1]: f[2] for f in facts if f[0] == "at_lander"},
        empty={f[1] for f in facts if f[0] == "empty"},
        full={f[1] for f in facts if f[0] == "full"},
        calibrated={(f[1], f[2]) for f in facts if f[0] == "calibrated"},
        have_rock={(f[1], f[2]) for f in facts if f[0] == "have_rock_analysis"},
        have_soil={(f[1], f[2]) for f in facts if f[0] == "have_soil_analysis"},
        have_image={(f[1], f[2], f[3]) for f in facts if f[0] == "have_image"},
        communicated_rock={f[1] for f in facts if f[0] == "communicated_rock_data"},
        communicated_soil={f[1] for f in facts if f[0] == "communicated_soil_data"},
        communicated_image={(f[1], f[2]) for f in facts if f[0] == "communicated_image_data"},
        rock_samples={f[1] for f in facts if f[0] == "at_rock_sample"},
        soil_samples={f[1] for f in facts if f[0] == "at_soil_sample"},
        static={f for f in facts if f[0] in static_names},
    )


def _parse_objects(problem: str) -> Dict[str, List[str]]:
    match = re.search(r"\(:objects\s+(.*?)\)\s*\(:init", problem, re.S)
    if not match:
        return {}
    tokens = match.group(1).split()
    objects: Dict[str, List[str]] = {}
    pending: List[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "-" and i + 1 < len(tokens):
            typ = tokens[i + 1]
            objects.setdefault(typ, []).extend(pending)
            pending = []
            i += 2
        else:
            pending.append(tokens[i])
            i += 1
    return {k: sorted(v, key=_name_key) for k, v in objects.items()}


def _parse_section_facts(problem: str, section_name: str) -> List[Fact]:
    start = problem.find(f"({section_name}")
    if start < 0:
        return []
    end = _matching_paren(problem, start)
    section = problem[start:end + 1]
    if section_name == ":goal":
        inner_start = section.find("(and")
        if inner_start >= 0:
            section = section[inner_start:]
    facts: List[Fact] = []
    for body in re.findall(r"\(([^()]+)\)", section):
        parts = tuple(body.split())
        if not parts or parts[0] in {":init", ":goal", "and"}:
            continue
        facts.append(parts)
    return facts


def _matching_paren(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced PDDL section")


def _name_key(name: str) -> Tuple[str, int]:
    match = re.match(r"([A-Za-z_]+)(\d+)$", name)
    if match:
        return (match.group(1), int(match.group(2)))
    return (name, -1)
