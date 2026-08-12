"""Natural-language parser for ACPBench hanoi instances."""

import re
from typing import Tuple

from architect.spec import HanoiGoals, HanoiState


DISK_RE = r"d\d+"
OBJ_RE = r"(?:d\d+|peg\d+)"


def parse_instance(context: str, inputs: str) -> Tuple[HanoiState, HanoiGoals]:
    count_match = re.search(r"There are\s+(\d+)\s+disks", context, re.I)
    n_disks = int(count_match.group(1)) if count_match else _max_disk_index(context + " " + inputs)
    disks = [f"d{i}" for i in range(1, n_disks + 1)]
    pegs = sorted(set(re.findall(r"peg\d+", context + " " + inputs)), key=_obj_key) or ["peg1", "peg2", "peg3"]

    on = {}
    stack_match = re.search(
        r"Currently,\s*The disks are stacked as follows:\s*(.*?)(?:\.\s*Following|\.\s*$)",
        context,
        re.I | re.S,
    )
    source = stack_match.group(1) if stack_match else context
    for child, support in re.findall(rf"({DISK_RE})\s+is\s+on\s+({OBJ_RE})", source, re.I):
        on[child.lower()] = support.lower()

    return HanoiState(disks=disks, pegs=pegs, on=on), parse_goals(inputs)


def parse_goals(inputs: str) -> HanoiGoals:
    goal_text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    facts = _split_goal_facts(goal_text)
    goals = HanoiGoals(on=set(), clear=set())

    for raw in facts:
        fact = raw.strip().rstrip(".?").lower()
        if not fact:
            continue
        objects = re.findall(OBJ_RE, fact)
        if not objects:
            continue

        if (
            "not obstructed" in fact
            or "clear" in fact
            or "no disk is placed on top of" in fact
            or "no disks are placed on top of" in fact
            or "no disk is on top of" in fact
            or "no disks are on top of" in fact
        ):
            goals.clear.add(objects[-1])
            continue

        if ("on top of" in fact or "above" in fact) and len(objects) >= 2:
            goals.on.add((objects[0], objects[1]))

    return goals


def _split_goal_facts(text: str):
    normalized = re.sub(r"\s+", " ", text)
    return re.split(r",\s*and\s+|\s+and\s+|,\s*", normalized)


def _max_disk_index(text: str) -> int:
    indexes = [int(d[1:]) for d in re.findall(DISK_RE, text, re.I)]
    return max(indexes) if indexes else 0


def _obj_key(obj: str):
    match = re.search(r"\d+", obj)
    return int(match.group()) if match else obj
