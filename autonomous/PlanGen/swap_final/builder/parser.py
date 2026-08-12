"""Natural-language parser for ACPBench swap instances."""

import re
from typing import Iterable, List, Tuple

from architect.spec import SwapGoals, SwapState


NAME_RE = r"[A-Za-z][A-Za-z0-9_-]*"


def parse_instance(context: str, inputs: str) -> Tuple[SwapState, SwapGoals]:
    agents = frozenset(_parse_named_list(context, "agents"))
    roles = frozenset(_parse_named_list(context, "items/roles"))
    assignments = _parse_assignments(context)
    if not agents:
        agents = frozenset(assignments)
    if not roles:
        roles = frozenset(assignments.values())
    return (
        SwapState(assignments=tuple(sorted(assignments.items())), agents=agents, roles=roles),
        parse_goals(inputs),
    )


def parse_goals(inputs: str) -> SwapGoals:
    goal_text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    atoms: List[Tuple[str, str]] = []
    for fact in _split_goal_facts(goal_text):
        match = re.search(rf"\b({NAME_RE})\s+is assigned\s+({NAME_RE})\b", fact, re.I)
        if match:
            atoms.append((match.group(1).lower(), match.group(2).lower()))
    return SwapGoals(tuple(atoms))


def _parse_named_list(context: str, label: str) -> List[str]:
    pattern = rf"There are\s+\d+\s+{re.escape(label)}:\s*(.*?)(?:\.|\n)"
    match = re.search(pattern, context, re.I | re.S)
    if not match:
        return []
    return [name.lower() for name in _split_list(match.group(1))]


def _parse_assignments(context: str) -> dict:
    assignments = {}
    for agent, role in re.findall(rf"\b({NAME_RE})\s+is assigned\s+({NAME_RE})\b", context, re.I):
        assignments[agent.lower()] = role.lower()
    return assignments


def _split_goal_facts(text: str) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text)
    return [part.strip(" .?") for part in re.split(r", and | and |, ", normalized) if part.strip()]


def _split_list(text: str) -> List[str]:
    text = text.replace(", and ", ", ").replace(" and ", ", ")
    return [part.strip().lower() for part in text.split(",") if part.strip()]
