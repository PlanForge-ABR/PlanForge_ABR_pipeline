"""Builder implementation of the architect's ferry methods."""

from copy import deepcopy
from typing import List, Optional, Tuple

from architect.spec import FerryGoals, FerryState, SolveResult


def construct_plan(initial: FerryState, goals: FerryGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)

    state = deepcopy(initial)
    plan: List[str] = []

    if goals.onboard:
        _prepare_empty_ferry(state, plan, goals.car_at)
        _put_car_onboard(state, plan, goals.onboard)
        if goals.ferry_at and state.ferry_at != goals.ferry_at:
            _sail(state, plan, goals.ferry_at)
        return _finish(initial, goals, plan)

    _prepare_empty_ferry(state, plan, goals.car_at)
    for car, target in sorted(goals.car_at.items(), key=lambda item: _object_key(item[0])):
        if state.car_at.get(car) == target:
            continue
        _transport_car(state, plan, car, target)

    if goals.empty and state.onboard is not None:
        _debark(state, plan, state.onboard, state.ferry_at)

    if goals.ferry_at and state.ferry_at != goals.ferry_at:
        _sail(state, plan, goals.ferry_at)

    return _finish(initial, goals, plan)


def validate_goal_consistency(state: FerryState, goals: FerryGoals) -> Tuple[bool, str]:
    if goals.car_location_conflict:
        return False, "a car cannot be at two distinct locations"
    if goals.ferry_location_conflict:
        return False, "the ferry cannot be at two distinct locations"
    if goals.onboard_conflict:
        return False, "the ferry can carry only one car"
    if goals.empty and goals.onboard:
        return False, "the ferry cannot be empty while a car is on board"
    if goals.onboard and goals.onboard in goals.car_at:
        return False, "a car cannot be both ashore at a location and on the ferry"
    for car in goals.car_at:
        if car not in state.cars:
            return False, f"unknown car {car}"
    if goals.onboard and goals.onboard not in state.cars:
        return False, f"unknown car {goals.onboard}"
    for loc in list(goals.car_at.values()) + ([goals.ferry_at] if goals.ferry_at else []):
        if loc not in state.locations:
            return False, f"unknown location {loc}"
    return True, ""


def simulate_plan(initial: FerryState, plan: List[str]) -> FerryState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        if parts[0] == "sail":
            if len(parts) == 3:
                if state.ferry_at != parts[1]:
                    raise ValueError(f"ferry is not at {parts[1]}")
                _sail(state, None, parts[2])
            else:
                raise ValueError(f"bad sail action: {action}")
        elif parts[0] == "board":
            _board(state, None, parts[1], parts[2])
        elif parts[0] == "debark":
            _debark(state, None, parts[1], parts[2])
        else:
            raise ValueError(f"unknown action: {action}")
    return state


def goals_hold(state: FerryState, goals: FerryGoals) -> bool:
    if goals.empty and state.onboard is not None:
        return False
    if goals.onboard and state.onboard != goals.onboard:
        return False
    if goals.ferry_at and state.ferry_at != goals.ferry_at:
        return False
    for car, loc in goals.car_at.items():
        if state.car_at.get(car) != loc:
            return False
    return True


def _finish(initial: FerryState, goals: FerryGoals, plan: List[str]) -> SolveResult:
    try:
        final_state = simulate_plan(initial, plan)
    except Exception as exc:
        return SolveResult(False, [], f"generated plan failed simulation: {exc}")
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "constructive planner could not satisfy the requested facts")
    return SolveResult(True, plan, "")


def _prepare_empty_ferry(state: FerryState, plan: List[str], car_targets) -> None:
    if state.onboard is None:
        return
    car = state.onboard
    target = car_targets.get(car, state.ferry_at)
    if state.ferry_at != target:
        _sail(state, plan, target)
    _debark(state, plan, car, target)


def _put_car_onboard(state: FerryState, plan: List[str], car: str) -> None:
    if state.onboard == car:
        return
    if state.onboard is not None:
        _debark(state, plan, state.onboard, state.ferry_at)
    car_loc = state.car_at[car]
    if state.ferry_at != car_loc:
        _sail(state, plan, car_loc)
    _board(state, plan, car, car_loc)


def _transport_car(state: FerryState, plan: List[str], car: str, target: str) -> None:
    if state.onboard == car:
        if state.ferry_at != target:
            _sail(state, plan, target)
        _debark(state, plan, car, target)
        return
    if state.onboard is not None:
        _debark(state, plan, state.onboard, state.ferry_at)
    start = state.car_at[car]
    if state.ferry_at != start:
        _sail(state, plan, start)
    _board(state, plan, car, start)
    if state.ferry_at != target:
        _sail(state, plan, target)
    _debark(state, plan, car, target)


def _sail(state: FerryState, plan: Optional[List[str]], to_loc: str) -> None:
    if state.ferry_at == to_loc:
        return
    from_loc = state.ferry_at
    state.ferry_at = to_loc
    if plan is not None:
        plan.append(f"sail {from_loc} {to_loc}")


def _board(state: FerryState, plan: Optional[List[str]], car: str, loc: str) -> None:
    if state.onboard is not None:
        raise ValueError("board requires an empty ferry")
    if state.ferry_at != loc:
        raise ValueError(f"ferry is not at {loc}")
    if state.car_at.get(car) != loc:
        raise ValueError(f"{car} is not at {loc}")
    del state.car_at[car]
    state.onboard = car
    if plan is not None:
        plan.append(f"board {car} {loc}")


def _debark(state: FerryState, plan: Optional[List[str]], car: str, loc: str) -> None:
    if state.onboard != car:
        raise ValueError(f"{car} is not on the ferry")
    if state.ferry_at != loc:
        raise ValueError(f"ferry is not at {loc}")
    state.onboard = None
    state.car_at[car] = loc
    if plan is not None:
        plan.append(f"debark {car} {loc}")


def _object_key(name: str):
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return (prefix, int(digits) if digits else -1)
