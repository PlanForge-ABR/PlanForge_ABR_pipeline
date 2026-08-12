"""Natural-language parser for ACPBench frogs_jumping instances."""

import re
from typing import Iterable, List, Tuple

from architect.spec import FrogGoals, FrogState


FROG_RE = r"[lr]\d+"
PAD_RE = r"p(\d+)"


def parse_instance(context: str, inputs: str) -> Tuple[FrogState, FrogGoals]:
    placements = _parse_placements(context)
    empty_pad = _parse_empty(context)
    max_pad = max([empty_pad, *placements.values()])
    board: List[str] = [""] * (max_pad + 1)
    board[empty_pad] = "_"
    for frog, pad in placements.items():
        board[pad] = frog
    for pad in range(1, max_pad + 1):
        if not board[pad]:
            board[pad] = "_"

    left = frozenset(frog for frog in placements if frog.startswith("l"))
    right = frozenset(frog for frog in placements if frog.startswith("r"))
    state = FrogState(
        board=tuple(board[1:]),
        positions=tuple(sorted(placements.items(), key=lambda item: _name_key(item[0]))),
        left_frogs=left,
        right_frogs=right,
    )
    return state, parse_goals(inputs)


def parse_goals(inputs: str) -> FrogGoals:
    text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    at_goals = []
    empty_goals = []

    for fact in _split_goal_facts(text):
        at_match = re.search(
            rf"\b({FROG_RE})\s+is\s+(?:on top of|above)\s+p(\d+)\b", fact, re.I
        )
        if at_match:
            at_goals.append((at_match.group(1).lower(), int(at_match.group(2))))
            continue

        empty_match = re.search(
            r"(?:no frog is at lily pad|p(\d+)\s+is\s+(?:empty|not occupied by any frog))\s*p?(\d+)?",
            fact,
            re.I,
        )
        if empty_match:
            pad = empty_match.group(1) or empty_match.group(2)
            if pad:
                empty_goals.append(int(pad))

    return FrogGoals(tuple(at_goals), tuple(empty_goals))


def _parse_placements(context: str) -> dict:
    return {
        frog.lower(): int(pad)
        for frog, pad in re.findall(rf"\b({FROG_RE})\s+is at\s+p(\d+)\b", context, re.I)
    }


def _parse_empty(context: str) -> int:
    match = re.search(r"The lily pad p(\d+) is empty", context, re.I)
    if not match:
        raise ValueError("could not find the empty lily pad")
    return int(match.group(1))


def _split_goal_facts(text: str) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text)
    return [part.strip(" .?") for part in re.split(r", and | and |, ", normalized) if part.strip()]


def _name_key(name: str) -> Tuple[str, int]:
    return name[0], int(name[1:])
