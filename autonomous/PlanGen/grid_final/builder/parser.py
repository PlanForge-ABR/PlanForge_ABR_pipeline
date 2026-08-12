"""Natural-language parser for ACPBench grid instances."""

import re
from typing import Iterable, List, Tuple

from architect.spec import GridGoals, GridState, GoalAtom


LOC_RE = r"f\d+-\d+f"
KEY_RE = r"key\d+-\d+"
SHAPE_RE = r"shape\d+"


def parse_instance(context: str, inputs: str) -> Tuple[GridState, GridGoals]:
    rows, cols = _parse_dimensions(context)

    key_shape = {
        key: shape
        for key, shape in re.findall(rf"Key\s+({KEY_RE})\s+is of shape\s+({SHAPE_RE})", context)
    }
    lock_shape = {
        loc: shape
        for loc, shape in re.findall(rf"({LOC_RE})\s+has\s+({SHAPE_RE})\s+shaped lock", context)
    }
    key_at = {
        key: loc
        for key, loc in re.findall(rf"Key\s+({KEY_RE})\s+is at position\s+({LOC_RE})", context)
    }

    robot_match = re.search(rf"robot is at position\s+({LOC_RE})", context, re.I)
    if not robot_match:
        raise ValueError("could not parse robot position")
    robot = robot_match.group(1)

    holding = None
    holding_match = re.search(rf"holding\s+({KEY_RE})", context, re.I)
    if holding_match:
        holding = holding_match.group(1)
        key_at.pop(holding, None)

    locked = set()
    locked_match = re.search(
        r"All the positions are open except the following:\s*(.*?)(?:\. Key|\. Currently|$)",
        context,
        re.I | re.S,
    )
    if locked_match:
        locked.update(re.findall(LOC_RE, locked_match.group(1)))

    return (
        GridState(
            rows=rows,
            cols=cols,
            robot=robot,
            holding=holding,
            key_at=tuple(sorted(key_at.items())),
            key_shape=tuple(sorted(key_shape.items())),
            lock_shape=tuple(sorted(lock_shape.items())),
            locked=frozenset(locked),
        ),
        parse_goals(inputs),
    )


def parse_goals(inputs: str) -> GridGoals:
    goal_text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    atoms: List[GoalAtom] = []
    for fact in _split_goal_facts(goal_text):
        low = fact.lower()
        locs = re.findall(LOC_RE, fact)
        keys = re.findall(KEY_RE, fact)
        loc = locs[-1] if locs else None
        key = keys[-1] if keys else None

        if "robot is at" in low and loc:
            atoms.append(("robot-at", loc, None))
        elif "robot is holding" in low and key:
            atoms.append(("holding", key, None))
        elif "arm is empty" in low:
            atoms.append(("arm-empty", "", None))
        elif low.startswith("key ") and key and loc:
            atoms.append(("key-at", key, loc))
        elif " is open" in low and loc:
            atoms.append(("open", loc, None))
        elif " is locked" in low and loc:
            atoms.append(("locked", loc, None))

    return GridGoals(frozenset(atoms))


def _parse_dimensions(context: str) -> Tuple[int, int]:
    match = re.search(r"grid size is\s+(\d+)x(\d+)", context, re.I)
    if match:
        return int(match.group(1)), int(match.group(2))
    locs = re.findall(LOC_RE, context)
    coords = [(int(a), int(b)) for a, b in re.findall(r"f(\d+)-(\d+)f", " ".join(locs))]
    if not coords:
        raise ValueError("could not parse grid dimensions")
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _split_goal_facts(text: str) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text)
    return [part.strip(" .?") for part in re.split(r", and | and |, ", normalized) if part.strip()]
