"""Parser for ACPBench depot instances."""

import re
from typing import Dict, List, Tuple

from architect.spec import DepotGoals, DepotState, Fact


FACT_RE = re.compile(r"\((at|available|clear|in|lifting|on)\s+([^()]*)\)")


def parse_instance(item: Dict[str, object]) -> Tuple[DepotState, DepotGoals]:
    problem = str(item.get("PDDL_problem") or "")
    if problem:
        return parse_pddl_problem(problem)
    raise ValueError("depot instances require PDDL_problem for exact structured parsing")


def parse_pddl_problem(problem: str) -> Tuple[DepotState, DepotGoals]:
    objects_text = _section(problem, ":objects", ":init")
    init_text = _section(problem, ":init", ":goal")
    goal_text = problem.split("(:goal", 1)[1] if "(:goal" in problem else ""

    typed = _parse_typed_objects(objects_text)
    crates = sorted(typed.get("crate", []), key=_object_key)
    pallets = sorted(typed.get("pallet", []), key=_object_key)
    hoists = sorted(typed.get("hoist", []), key=_object_key)
    trucks = sorted(typed.get("truck", []), key=_object_key)
    places = sorted(typed.get("depot", []) + typed.get("distributor", []), key=_place_key)

    state = DepotState(crates=crates, pallets=pallets, hoists=hoists, trucks=trucks, places=places)
    for fact in _facts(init_text):
        pred = fact[0]
        if pred == "at":
            state.at[fact[1]] = fact[2]
        elif pred == "available":
            state.available.add(fact[1])
        elif pred == "clear":
            state.clear.add(fact[1])
        elif pred == "on":
            state.on[fact[1]] = fact[2]
        elif pred == "in":
            state.in_truck[fact[1]] = fact[2]
        elif pred == "lifting":
            state.lifting[fact[1]] = fact[2]

    goals = DepotGoals(_facts(goal_text))
    return state, goals


def _section(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    return text.split(f"({start}", 1)[1].split(f"({end}", 1)[0]


def _parse_typed_objects(text: str) -> Dict[str, List[str]]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|-", text)
    typed: Dict[str, List[str]] = {}
    pending: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-" and i + 1 < len(tokens):
            typ = tokens[i + 1]
            typed.setdefault(typ, []).extend(pending)
            pending = []
            i += 2
        else:
            pending.append(tok)
            i += 1
    return typed


def _facts(text: str) -> List[Fact]:
    facts: List[Fact] = []
    for match in FACT_RE.finditer(text):
        pred = match.group(1)
        args = tuple(match.group(2).split())
        facts.append((pred, *args))
    return facts


def _object_key(name: str):
    match = re.match(r"([A-Za-z_]+)(\d+)$", name)
    if not match:
        return (name, -1)
    return (match.group(1), int(match.group(2)))


def _place_key(name: str):
    if name.startswith("depot"):
        group = 0
    elif name.startswith("distributor"):
        group = 1
    else:
        group = 2
    number = re.search(r"(\d+)$", name)
    return (group, int(number.group(1)) if number else -1, name)
