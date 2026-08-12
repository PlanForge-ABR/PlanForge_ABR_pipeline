"""Instance parser for satellite benchmark records."""

import re
from typing import Dict, Iterable, List, Tuple

from architect.spec import Fact, SatelliteState


FACT_RE = re.compile(r"\(([a-zA-Z_][\w-]*)([^()]*)\)")


def parse_instance(item: Dict[str, object]) -> Tuple[SatelliteState, List[Fact]]:
    """Parse one dataset item into the architect's state and goal schema."""
    problem = str(item.get("PDDL_problem", ""))
    state = _parse_problem(problem)
    goals = _parse_goals(str(item.get("inputs", "")), problem)
    return state, goals


def _parse_problem(problem: str) -> SatelliteState:
    state = SatelliteState()
    _parse_objects(problem, state)
    init_text = _section(problem, ":init", ":goal")
    for pred, args in _facts(init_text):
        if pred == "on_board" and len(args) == 2:
            state.on_board[args[0]] = args[1]
        elif pred == "supports" and len(args) == 2:
            state.supports.setdefault(args[0], set()).add(args[1])
        elif pred == "calibration_target" and len(args) == 2:
            state.calibration_targets.setdefault(args[0], []).append(args[1])
        elif pred == "pointing" and len(args) == 2:
            state.pointing[args[0]] = args[1]
        elif pred == "power_avail" and len(args) == 1:
            state.power_avail.add(args[0])
        elif pred == "power_on" and len(args) == 1:
            state.power_on.add(args[0])
        elif pred == "calibrated" and len(args) == 1:
            state.calibrated.add(args[0])
        elif pred == "have_image" and len(args) == 2:
            state.have_images.add((args[0], args[1]))
    return state


def _parse_objects(problem: str, state: SatelliteState) -> None:
    match = re.search(r"\(:objects(.*?)\)\s*\(:init", problem, flags=re.S)
    if not match:
        return
    tokens = match.group(1).split()
    pending: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "-" and i + 1 < len(tokens):
            typ = tokens[i + 1]
            target = {
                "direction": state.directions,
                "instrument": state.instruments,
                "mode": state.modes,
                "satellite": state.satellites,
            }.get(typ)
            if target is not None:
                target.update(pending)
            pending = []
            i += 2
        else:
            pending.append(token)
            i += 1


def _parse_goals(inputs: str, problem: str) -> List[Fact]:
    goals: List[Fact] = []
    for mode, direction in re.findall(r"A ([\w-]+) mode image of target ([\w-]+) is available", inputs):
        goals.append(("have_image", direction, mode))
    for satellite, direction in re.findall(r"Satellite (satellite\d+) is pointing to ([\w-]+)", inputs):
        goals.append(("pointing", satellite, direction))
    for instrument in re.findall(r"Following instruments are powered on: ([\w-]+)", inputs):
        goals.append(("power_on", instrument))
    for instrument in re.findall(r"Following instruments are calibrated: ([\w-]+)", inputs):
        goals.append(("calibrated", instrument))

    if goals:
        return goals

    goal_text = _section(problem, ":goal", None)
    for pred, args in _facts(goal_text):
        goals.append(tuple([pred, *args]))
    return goals


def _section(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(f"({start_marker}")
    if start < 0:
        return ""
    if end_marker is None:
        return text[start:]
    end = text.find(f"({end_marker}", start)
    return text[start:end if end >= 0 else len(text)]


def _facts(text: str) -> Iterable[Tuple[str, List[str]]]:
    for pred, rest in FACT_RE.findall(text):
        if pred in {"and", "define", "problem", "domain", ":objects", ":init", ":goal"}:
            continue
        yield pred.replace("-", "_"), rest.split()
