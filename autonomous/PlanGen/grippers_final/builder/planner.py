"""Builder implementation of the architect's grippers methods."""

from copy import deepcopy
from typing import List, Optional, Tuple

from architect.spec import GrippersGoals, GrippersState, SolveResult


ROBOT = "robot1"
GRIPPERS = ("left1", "right1")


def construct_plan(initial: GrippersState, goals: GrippersGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)

    state = deepcopy(initial)
    plan: List[str] = []

    for gripper, ball in list(state.carrying.items()):
        if gripper in goals.carrying and goals.carrying[gripper] == ball:
            continue
        if ball in goals.ball_at:
            _move_robot(state, plan, goals.ball_at[ball])
        _drop(state, plan, ball, state.robot_at[ROBOT], gripper)

    for ball, target_room in sorted(goals.ball_at.items(), key=lambda item: _object_key(item[0])):
        if state.ball_at.get(ball) == target_room:
            continue
        _deliver_ball(state, plan, ball, target_room)

    for gripper in sorted(goals.free, key=_object_key):
        if gripper in state.carrying:
            _drop(state, plan, state.carrying[gripper], state.robot_at[ROBOT], gripper)

    for gripper, ball in sorted(goals.carrying.items(), key=lambda item: _object_key(item[0])):
        if state.carrying.get(gripper) == ball:
            continue
        if gripper in state.carrying:
            _drop(state, plan, state.carrying[gripper], state.robot_at[ROBOT], gripper)
        holder = _holder_of(state, ball)
        if holder and holder != gripper:
            _drop(state, plan, ball, state.robot_at[ROBOT], holder)
        _pick_into_gripper(state, plan, ball, gripper)

    final_room = goals.robot_at.get(ROBOT)
    if final_room:
        _move_robot(state, plan, final_room)

    return _finish(initial, goals, plan)


def validate_goal_consistency(state: GrippersState, goals: GrippersGoals) -> Tuple[bool, str]:
    if goals.ball_location_conflict:
        return False, "a ball cannot be at two distinct rooms"
    if goals.robot_location_conflict:
        return False, "a robot cannot be at two distinct rooms"
    if goals.gripper_carry_conflict:
        return False, "one gripper cannot carry two distinct balls"
    if goals.ball_carry_conflict:
        return False, "one ball cannot be carried by two grippers"
    if set(goals.free) & set(goals.carrying):
        return False, "a gripper cannot be both free and carrying a ball"

    carried_goal_balls = set(goals.carrying.values())
    located_goal_balls = set(goals.ball_at)
    if carried_goal_balls & located_goal_balls:
        return False, "a ball cannot be both carried and located in a room"

    if len(carried_goal_balls) != len(goals.carrying):
        return False, "two grippers cannot carry the same ball"
    if len(goals.carrying) > len(state.grippers):
        return False, "more carry goals than available grippers"

    for ball in list(goals.ball_at) + list(goals.carrying.values()):
        if ball not in state.balls:
            return False, f"unknown ball {ball}"
    for room in list(goals.ball_at.values()) + list(goals.robot_at.values()):
        if room not in state.rooms:
            return False, f"unknown room {room}"
    for gripper in list(goals.carrying) + list(goals.free):
        if gripper not in state.grippers:
            return False, f"unknown gripper {gripper}"
    for robot in goals.robot_at:
        if robot not in state.robots:
            return False, f"unknown robot {robot}"
    return True, ""


def simulate_plan(initial: GrippersState, plan: List[str]) -> GrippersState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        if not parts:
            continue
        if parts[0] == "move" and len(parts) == 4:
            robot, from_room, to_room = parts[1:]
            if state.robot_at.get(robot) != from_room:
                raise ValueError(f"{robot} is not at {from_room}")
            state.robot_at[robot] = to_room
        elif parts[0] == "pick" and len(parts) == 5:
            robot, ball, room, gripper = parts[1:]
            _pick(state, None, robot, ball, room, gripper)
        elif parts[0] == "drop" and len(parts) == 5:
            robot, ball, room, gripper = parts[1:]
            _drop_checked(state, None, robot, ball, room, gripper)
        else:
            raise ValueError(f"bad action: {action}")
    return state


def goals_hold(state: GrippersState, goals: GrippersGoals) -> bool:
    for robot, room in goals.robot_at.items():
        if state.robot_at.get(robot) != room:
            return False
    for ball, room in goals.ball_at.items():
        if state.ball_at.get(ball) != room:
            return False
    for gripper, ball in goals.carrying.items():
        if state.carrying.get(gripper) != ball:
            return False
    for gripper in goals.free:
        if gripper not in state.free:
            return False
    return True


def _finish(initial: GrippersState, goals: GrippersGoals, plan: List[str]) -> SolveResult:
    try:
        final_state = simulate_plan(initial, plan)
    except Exception as exc:
        return SolveResult(False, [], f"generated plan failed simulation: {exc}")
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "constructive planner could not satisfy the requested facts")
    return SolveResult(True, plan, "")


def _deliver_ball(state: GrippersState, plan: List[str], ball: str, target_room: str) -> None:
    holder = _holder_of(state, ball)
    if holder:
        _move_robot(state, plan, target_room)
        _drop(state, plan, ball, target_room, holder)
        return

    source_room = state.ball_at[ball]
    gripper = _free_gripper(state)
    if gripper is None:
        gripper = next(g for g in GRIPPERS if g not in state.carrying or state.carrying[g] != ball)
        _drop(state, plan, state.carrying[gripper], state.robot_at[ROBOT], gripper)
    _move_robot(state, plan, source_room)
    _pick(state, plan, ROBOT, ball, source_room, gripper)
    _move_robot(state, plan, target_room)
    _drop(state, plan, ball, target_room, gripper)


def _pick_into_gripper(state: GrippersState, plan: List[str], ball: str, gripper: str) -> None:
    source_room = state.ball_at[ball]
    _move_robot(state, plan, source_room)
    _pick(state, plan, ROBOT, ball, source_room, gripper)


def _move_robot(state: GrippersState, plan: List[str], to_room: str) -> None:
    from_room = state.robot_at[ROBOT]
    if from_room == to_room:
        return
    state.robot_at[ROBOT] = to_room
    plan.append(f"move {ROBOT} {from_room} {to_room}")


def _pick(state: GrippersState, plan: Optional[List[str]], robot: str, ball: str, room: str, gripper: str) -> None:
    if state.robot_at.get(robot) != room:
        raise ValueError(f"{robot} is not at {room}")
    if state.ball_at.get(ball) != room:
        raise ValueError(f"{ball} is not at {room}")
    if gripper not in state.free:
        raise ValueError(f"{gripper} is not free")
    del state.ball_at[ball]
    state.free.remove(gripper)
    state.carrying[gripper] = ball
    if plan is not None:
        plan.append(f"pick {robot} {ball} {room} {gripper}")


def _drop(state: GrippersState, plan: List[str], ball: str, room: str, gripper: str) -> None:
    _drop_checked(state, plan, ROBOT, ball, room, gripper)


def _drop_checked(
    state: GrippersState, plan: Optional[List[str]], robot: str, ball: str, room: str, gripper: str
) -> None:
    if state.robot_at.get(robot) != room:
        raise ValueError(f"{robot} is not at {room}")
    if state.carrying.get(gripper) != ball:
        raise ValueError(f"{gripper} is not carrying {ball}")
    del state.carrying[gripper]
    state.free.add(gripper)
    state.ball_at[ball] = room
    if plan is not None:
        plan.append(f"drop {robot} {ball} {room} {gripper}")


def _free_gripper(state: GrippersState) -> Optional[str]:
    for gripper in GRIPPERS:
        if gripper in state.free:
            return gripper
    return None


def _holder_of(state: GrippersState, ball: str) -> Optional[str]:
    for gripper, carried_ball in state.carrying.items():
        if carried_ball == ball:
            return gripper
    return None


def _object_key(name: str):
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return (prefix, int(digits) if digits else -1)
