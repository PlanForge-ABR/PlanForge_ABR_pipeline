"""Natural-language parser for ACPBench blocksworld instances."""

import re
from typing import Tuple

from architect.spec import BWGoals, BWState


BLOCK_RE = r"block_\d+"


def _blocks(text: str):
    return re.findall(BLOCK_RE, text)


def parse_instance(context: str, inputs: str) -> Tuple[BWState, BWGoals]:
    blocks = [f"block_{i}" for i in range(1, _parse_count(context) + 1)]

    holding = None
    if re.search(r"robotic arm is empty", context, re.I):
        holding = None
    else:
        match = re.search(r"robotic arm is holding (block_\d+)", context, re.I)
        if match:
            holding = match.group(1)

    on = {child: parent for child, parent in re.findall(rf"({BLOCK_RE}) is on ({BLOCK_RE})", context)}
    mentioned_table = _parse_table_blocks(context)
    ontable = set(mentioned_table)

    for block in _blocks(context + " " + inputs):
        if block not in blocks:
            blocks.append(block)

    goals = parse_goals(inputs)
    return BWState(blocks=blocks, on=on, ontable=ontable, holding=holding), goals


def _parse_count(context: str) -> int:
    match = re.search(r"There are (\d+) blocks", context)
    if not match:
        found = [int(b.split("_")[1]) for b in _blocks(context)]
        return max(found) if found else 0
    return int(match.group(1))


def _parse_table_blocks(context: str):
    match = re.search(
        r"The following block\(s\) (?:are|is) on the table:(.*?)(?:\. The following|$)",
        context,
        re.S,
    )
    if not match:
        return []
    return _blocks(match.group(1))


def parse_goals(inputs: str) -> BWGoals:
    goal_text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    facts = _split_goal_facts(goal_text)
    goals = BWGoals(on=set(), clear=set(), ontable=set())

    for fact in facts:
        fact = fact.strip().rstrip(".?")
        blocks = _blocks(fact)
        low = fact.lower()

        if "not holding anything" in low or "robotic arm is empty" in low:
            goals.handempty = True
            continue

        if ("currently being held" in low or "robotic arm is holding" in low) and blocks:
            held = blocks[-1]
            if goals.holding is not None and goals.holding != held:
                goals.holding_conflict = True
            goals.holding = held
            continue

        if "on top of" in low and len(blocks) >= 2:
            goals.on.add((blocks[0], blocks[1]))
            continue

        if "situated above" in low and len(blocks) >= 2:
            goals.on.add((blocks[0], blocks[1]))
            continue

        if "situated under" in low and len(blocks) >= 2:
            goals.on.add((blocks[1], blocks[0]))
            continue

        if (
            " is clear" in low
            or "not obstructed" in low
            or "no blocks are placed on top" in low
        ) and blocks:
            goals.clear.add(blocks[-1])
            continue

        if "located on the table" in low and blocks:
            goals.ontable.add(blocks[-1])

    return goals


def _split_goal_facts(text: str):
    normalized = re.sub(r"\s+", " ", text)
    return re.split(r", and | and |, ", normalized)
