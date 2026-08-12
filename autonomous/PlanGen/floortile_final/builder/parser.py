"""Structured parser for ACPBench floortile instances."""

import re
from typing import Dict, List, Tuple

from architect.spec import Fact, FloorTileGoals, FloorTileState


FACT_RE = re.compile(
    r"\((available-color|clear|down|left|painted|right|robot-at|robot-has|up)\s+([^()]*)\)"
)


def parse_instance(item: Dict[str, object]) -> Tuple[FloorTileState, FloorTileGoals]:
    problem = str(item.get("PDDL_problem") or "")
    if not problem:
        raise ValueError("floortile instances require PDDL_problem for exact structured parsing")
    return parse_pddl_problem(problem)


def parse_pddl_problem(problem: str) -> Tuple[FloorTileState, FloorTileGoals]:
    objects_text = _section(problem, ":objects", ":init")
    init_text = _section(problem, ":init", ":goal")
    goal_text = problem.split("(:goal", 1)[1] if "(:goal" in problem else ""

    typed = _parse_typed_objects(objects_text)
    colors = sorted(typed.get("color", []), key=_name_key)
    robots = sorted(typed.get("robot", []), key=_name_key)
    tiles = sorted(typed.get("tile", []), key=_name_key)
    state = FloorTileState(
        colors=colors,
        robots=robots,
        tiles=tiles,
        move_edges={tile: {} for tile in tiles},
        robot_at={},
        robot_has={},
    )

    for fact in _facts(init_text):
        pred = fact[0]
        if pred in {"up", "down", "left", "right"}:
            state.move_edges.setdefault(fact[2], {})[pred] = fact[1]
        elif pred == "clear":
            state.clear.add(fact[1])
        elif pred == "painted":
            state.painted[fact[1]] = fact[2]
        elif pred == "robot-at":
            state.robot_at[fact[1]] = fact[2]
        elif pred == "robot-has":
            state.robot_has[fact[1]] = fact[2]

    return state, FloorTileGoals(_facts(goal_text))


def _section(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    return text.split(f"({start}", 1)[1].split(f"({end}", 1)[0]


def _facts(text: str) -> List[Fact]:
    return [(m.group(1), *tuple(m.group(2).split())) for m in FACT_RE.finditer(text)]


def _parse_typed_objects(text: str) -> Dict[str, List[str]]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|-", text)
    typed: Dict[str, List[str]] = {}
    pending: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-" and i + 1 < len(tokens):
            typed.setdefault(tokens[i + 1], []).extend(pending)
            pending = []
            i += 2
        else:
            pending.append(tok)
            i += 1
    return typed


def _name_key(name: str):
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return (prefix, int(digits) if digits else -1, name)
