"""Natural-language parser for ACPBench goldminer instances."""

import re
from typing import FrozenSet, Iterable, List, Tuple

from architect.spec import GMGoals, GMState, GoalAtom


LOC_RE = r"f\d+-\d+f"


def parse_instance(context: str, inputs: str) -> Tuple[GMState, GMGoals]:
    rows, cols = _parse_dimensions(context)
    all_locs = {f"f{r}-{c}f" for r in range(rows) for c in range(cols)}
    soft = set(_parse_location_list(context, r"soft rock"))
    hard = set(_parse_location_list(context, r"hard rock"))
    gold = set(re.findall(r"The gold is at (f\d+-\d+f) location", context))

    robot_match = re.search(r"robot is at position (f\d+-\d+f)", context, re.I)
    if not robot_match:
        raise ValueError("could not parse robot position")
    robot = robot_match.group(1)

    bomb_match = re.search(r"Bomb supply is available at (f\d+-\d+f) location", context, re.I)
    bomb_at = bomb_match.group(1) if bomb_match else "f0-0f"

    laser_match = re.search(r"The laser is at (f\d+-\d+f) location", context, re.I)
    laser_at = laser_match.group(1) if laser_match else None

    holding = "empty"
    if re.search(r"holding a laser", context, re.I):
        holding = "laser"
    elif re.search(r"holding a bomb", context, re.I):
        holding = "bomb"
    elif re.search(r"holding gold", context, re.I):
        holding = "gold"

    clear = all_locs - soft - hard
    goals = parse_goals(inputs)
    return (
        GMState(
            rows=rows,
            cols=cols,
            robot=robot,
            holding=holding,
            bomb_at=bomb_at,
            laser_at=laser_at,
            clear=frozenset(clear),
            soft=frozenset(soft),
            hard=frozenset(hard),
            gold=frozenset(gold),
        ),
        goals,
    )


def parse_goals(inputs: str) -> GMGoals:
    goal_text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    chunks = _split_goal_facts(goal_text)
    atoms: List[GoalAtom] = []
    for fact in chunks:
        low = fact.lower()
        locs = re.findall(LOC_RE, fact)
        loc = locs[-1] if locs else None

        if "holding gold" in low:
            atoms.append(("holds-gold", None))
        elif "holding laser" in low:
            atoms.append(("holds-laser", None))
        elif "holding bomb" in low:
            atoms.append(("holds-bomb", None))
        elif "arm is empty" in low or "not holding" in low:
            atoms.append(("arm-empty", None))
        elif "robot is at position" in low and loc:
            atoms.append(("robot-at", loc))
        elif "soft rock at" in low and loc:
            atoms.append(("soft-rock-at", loc))
        elif "hard rock at" in low and loc:
            atoms.append(("hard-rock-at", loc))
        elif "gold is at" in low and loc:
            atoms.append(("gold-at", loc))
        elif "laser is at" in low and loc:
            atoms.append(("laser-at", loc))
        elif "clear" in low and loc:
            atoms.append(("clear", loc))

    return GMGoals(frozenset(atoms))


def _parse_dimensions(context: str) -> Tuple[int, int]:
    match = re.search(r"The (\d+)x(\d+) grid", context)
    if match:
        return int(match.group(1)), int(match.group(2))
    locs = re.findall(LOC_RE, context)
    if not locs:
        raise ValueError("could not parse grid dimensions")
    coords = [(int(a), int(b)) for a, b in re.findall(r"f(\d+)-(\d+)f", " ".join(locs))]
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _parse_location_list(context: str, rock_name: str) -> List[str]:
    pattern = rf"The following locations have {rock_name}: (.*?)(?:\. The|\. Currently|$)"
    match = re.search(pattern, context, re.I | re.S)
    return re.findall(LOC_RE, match.group(1)) if match else []


def _split_goal_facts(text: str) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text)
    return [part.strip(" .?") for part in re.split(r", and | and |, ", normalized) if part.strip()]
