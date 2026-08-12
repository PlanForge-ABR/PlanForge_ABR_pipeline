"""Natural-language parser for ACPBench grippers instances."""

import re
from typing import Dict, List, Tuple

from architect.spec import GrippersGoals, GrippersState


BALL_RE = r"ball\d+"
ROOM_RE = r"room\d+"
ROBOT_RE = r"robot\d+"
GRIPPER_WORD_RE = r"(left|right)(?:\d+)?"


def parse_instance(context: str, inputs: str) -> Tuple[GrippersState, GrippersGoals]:
    robot_count = _parse_count(context, "robot")
    room_count = _parse_count(context, "rooms")
    ball_count = _parse_count(context, "balls")

    robots = [f"robot{i}" for i in range(1, robot_count + 1)] or ["robot1"]
    rooms = [f"room{i}" for i in range(1, room_count + 1)]
    balls = [f"ball{i}" for i in range(1, ball_count + 1)]
    grippers = ["left1", "right1"]

    for name in sorted(set(re.findall(ROBOT_RE, context + " " + inputs)), key=_object_key):
        if name not in robots:
            robots.append(name)
    for name in sorted(set(re.findall(ROOM_RE, context + " " + inputs)), key=_object_key):
        if name not in rooms:
            rooms.append(name)
    for name in sorted(set(re.findall(BALL_RE, context + " " + inputs)), key=_object_key):
        if name not in balls:
            balls.append(name)

    robot_at = _parse_robot_location(context)
    ball_at = _parse_ball_locations(context)
    carrying = _parse_carrying_context(context)
    free = _parse_free_grippers_context(context, carrying)
    goals = parse_goals(inputs)

    return (
        GrippersState(
            robots=robots,
            rooms=rooms,
            balls=balls,
            grippers=grippers,
            robot_at=robot_at,
            ball_at=ball_at,
            carrying=carrying,
            free=free,
        ),
        goals,
    )


def parse_goals(inputs: str) -> GrippersGoals:
    text = inputs.split("holds:", 1)[-1].strip().rstrip("?")
    facts = _split_goal_facts(text)
    goals = GrippersGoals()
    seen_robot_locations: Dict[str, List[str]] = {}
    seen_ball_locations: Dict[str, List[str]] = {}
    seen_gripper_carries: Dict[str, List[str]] = {}
    seen_ball_carries: Dict[str, List[str]] = {}

    for raw_fact in facts:
        fact = raw_fact.strip().rstrip(".?")
        if not fact:
            continue

        robot_loc = re.search(
            rf"Robot ({ROBOT_RE}) is (?:in room|at) ({ROOM_RE})(?: location)?",
            fact,
            re.I,
        )
        if robot_loc:
            robot, room = robot_loc.groups()
            seen_robot_locations.setdefault(robot, []).append(room)
            goals.robot_at[robot] = room
            continue

        ball_loc = re.search(
            rf"Ball ({BALL_RE}) is (?:in room|at) ({ROOM_RE})(?: location)?",
            fact,
            re.I,
        )
        if ball_loc:
            ball, room = ball_loc.groups()
            seen_ball_locations.setdefault(ball, []).append(room)
            goals.ball_at[ball] = room
            continue

        carry = re.search(
            rf"Robot ({ROBOT_RE}) is carrying the ball ({BALL_RE}) in the {GRIPPER_WORD_RE} gripper",
            fact,
            re.I,
        )
        if carry:
            _robot, ball, side = carry.groups()
            gripper = _canonical_gripper(side)
            seen_gripper_carries.setdefault(gripper, []).append(ball)
            seen_ball_carries.setdefault(ball, []).append(gripper)
            goals.carrying[gripper] = ball
            continue

        free = re.search(
            rf"The {GRIPPER_WORD_RE} gripper of robot ({ROBOT_RE}) is free",
            fact,
            re.I,
        )
        if free:
            side, _robot = free.groups()
            goals.free.add(_canonical_gripper(side))

    goals.robot_location_conflict = any(len(set(values)) > 1 for values in seen_robot_locations.values())
    goals.ball_location_conflict = any(len(set(values)) > 1 for values in seen_ball_locations.values())
    goals.gripper_carry_conflict = any(len(set(values)) > 1 for values in seen_gripper_carries.values())
    goals.ball_carry_conflict = any(len(set(values)) > 1 for values in seen_ball_carries.values())
    return goals


def _parse_count(context: str, noun: str) -> int:
    match = re.search(rf"There (?:is|are) (\d+) {noun}", context, re.I)
    return int(match.group(1)) if match else 0


def _parse_robot_location(context: str) -> Dict[str, str]:
    match = re.search(rf"robot ({ROBOT_RE}) is at ({ROOM_RE})", context, re.I)
    if not match:
        raise ValueError("could not parse robot location")
    robot, room = match.groups()
    return {robot: room}


def _parse_carrying_context(context: str) -> Dict[str, str]:
    carrying: Dict[str, str] = {}
    for side, ball in re.findall(rf"{GRIPPER_WORD_RE} gripper is carrying the ball ({BALL_RE})", context, re.I):
        carrying[_canonical_gripper(side)] = ball
    return carrying


def _parse_free_grippers_context(context: str, carrying: Dict[str, str]):
    if "both grippers are free" in context.lower():
        return {"left1", "right1"}
    free = set()
    for side in re.findall(rf"{GRIPPER_WORD_RE} gripper is free", context, re.I):
        free.add(_canonical_gripper(side))
    return free | ({g for g in ("left1", "right1") if g not in carrying and g not in free} - set())


def _parse_ball_locations(context: str) -> Dict[str, str]:
    if "Additionally," not in context:
        return {}
    text = context.split("Additionally,", 1)[1].strip().rstrip(".")
    ball_at: Dict[str, str] = {}
    pattern = rf"(.+?)\s+(?:are|is) at ({ROOM_RE})(?:,|\.|$)"
    for balls_text, room in re.findall(pattern, text, re.I):
        for ball in re.findall(BALL_RE, balls_text):
            ball_at[ball] = room
    return ball_at


def _split_goal_facts(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text)
    return re.split(r", and | and |, ", normalized)


def _canonical_gripper(side: str) -> str:
    return "left1" if side.lower().startswith("left") else "right1"


def _object_key(name: str):
    match = re.match(r"([A-Za-z_]+)(\d+)$", name)
    if not match:
        return (name, -1)
    return (match.group(1), int(match.group(2)))
