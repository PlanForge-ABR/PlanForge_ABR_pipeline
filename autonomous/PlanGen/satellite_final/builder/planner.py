"""Constructive satellite planner and validator."""

from copy import deepcopy
from typing import Iterable, List, Optional, Sequence

from architect.spec import Fact, PlanResult, SatelliteState


def construct_plan(state: SatelliteState, goals: Sequence[Fact]) -> PlanResult:
    working = deepcopy(state)
    plan: List[str] = []

    conflict = _pointing_conflict(goals)
    if conflict:
        return PlanResult(False, [], conflict)

    final_power = {g[1] for g in goals if g[0] == "power_on"}
    final_calibrated = {g[1] for g in goals if g[0] == "calibrated"}
    final_pointing = {(g[1], g[2]) for g in goals if g[0] == "pointing"}

    for goal in goals:
        if goal[0] == "have_image" and (goal[1], goal[2]) not in working.have_images:
            if not _achieve_image(working, goal[1], goal[2], plan):
                return PlanResult(False, [], f"no usable powered/calibratable instrument supports {goal[2]}")

    for instrument in sorted(final_calibrated):
        if instrument not in working.calibrated:
            if not _ensure_calibrated(working, instrument, plan):
                return PlanResult(False, [], f"cannot calibrate {instrument}")

    for satellite, direction in sorted(final_pointing):
        if working.pointing.get(satellite) != direction:
            _turn(working, satellite, direction, plan)

    for instrument in sorted(final_power):
        satellite = working.on_board.get(instrument)
        if satellite is None:
            return PlanResult(False, [], f"{instrument} is not on board any satellite")
        if instrument not in working.power_on:
            _make_power_available(working, satellite, plan, keep=None)
            if satellite not in working.power_avail:
                return PlanResult(False, [], f"no power available for {instrument}")
            _switch_on(working, instrument, satellite, plan)

    if goals_hold(working, goals):
        return PlanResult(True, plan, "constructed by satellite resource planner")
    return PlanResult(False, [], "constructed plan did not satisfy all goals")


def goals_hold(state: SatelliteState, goals: Iterable[Fact]) -> bool:
    for goal in goals:
        pred = goal[0]
        if pred == "pointing" and state.pointing.get(goal[1]) != goal[2]:
            return False
        if pred == "have_image" and (goal[1], goal[2]) not in state.have_images:
            return False
        if pred == "power_on" and goal[1] not in state.power_on:
            return False
        if pred == "calibrated" and goal[1] not in state.calibrated:
            return False
    return True


def simulate_plan(initial: SatelliteState, plan: Sequence[str]) -> SatelliteState:
    state = deepcopy(initial)
    for step in plan:
        parts = step.split()
        if not parts:
            continue
        action = parts[0]
        if action == "turn_to" and len(parts) == 4:
            sat, new, old = parts[1], parts[2], parts[3]
            if state.pointing.get(sat) != old:
                raise ValueError(f"invalid turn_to precondition: {step}")
            state.pointing[sat] = new
        elif action == "switch_on" and len(parts) == 3:
            inst, sat = parts[1], parts[2]
            if state.on_board.get(inst) != sat or sat not in state.power_avail:
                raise ValueError(f"invalid switch_on precondition: {step}")
            state.power_on.add(inst)
            state.calibrated.discard(inst)
            state.power_avail.discard(sat)
        elif action == "switch_off" and len(parts) == 3:
            inst, sat = parts[1], parts[2]
            if state.on_board.get(inst) != sat or inst not in state.power_on:
                raise ValueError(f"invalid switch_off precondition: {step}")
            state.power_on.discard(inst)
            state.power_avail.add(sat)
        elif action == "calibrate" and len(parts) == 4:
            sat, inst, direction = parts[1], parts[2], parts[3]
            if (
                state.on_board.get(inst) != sat
                or direction not in state.calibration_targets.get(inst, [])
                or state.pointing.get(sat) != direction
                or inst not in state.power_on
            ):
                raise ValueError(f"invalid calibrate precondition: {step}")
            state.calibrated.add(inst)
        elif action == "take_image" and len(parts) == 5:
            sat, direction, inst, mode = parts[1], parts[2], parts[3], parts[4]
            if (
                inst not in state.calibrated
                or state.on_board.get(inst) != sat
                or mode not in state.supports.get(inst, set())
                or inst not in state.power_on
                or state.pointing.get(sat) != direction
            ):
                raise ValueError(f"invalid take_image precondition: {step}")
            state.have_images.add((direction, mode))
        else:
            raise ValueError(f"unknown action: {step}")
    return state


def _achieve_image(state: SatelliteState, direction: str, mode: str, plan: List[str]) -> bool:
    for instrument in _candidate_instruments(state, mode):
        trial = deepcopy(state)
        trial_plan: List[str] = []
        if not _ensure_calibrated(trial, instrument, trial_plan):
            continue
        satellite = trial.on_board[instrument]
        if trial.pointing.get(satellite) != direction:
            _turn(trial, satellite, direction, trial_plan)
        trial.have_images.add((direction, mode))
        trial_plan.append(f"take_image {satellite} {direction} {instrument} {mode}")
        _copy_state(trial, state)
        plan.extend(trial_plan)
        return True
    return False


def _candidate_instruments(state: SatelliteState, mode: str) -> List[str]:
    powered = [i for i in state.power_on if mode in state.supports.get(i, set())]
    unpowered = [i for i in state.instruments if i not in state.power_on and mode in state.supports.get(i, set())]
    return sorted(powered) + sorted(unpowered)


def _ensure_calibrated(state: SatelliteState, instrument: str, plan: List[str]) -> bool:
    satellite = state.on_board.get(instrument)
    if satellite is None:
        return False
    if instrument not in state.power_on:
        _make_power_available(state, satellite, plan, keep=None)
        if satellite not in state.power_avail:
            return False
        _switch_on(state, instrument, satellite, plan)
    target = _best_calibration_target(state, satellite, instrument)
    if target is None:
        return False
    if state.pointing.get(satellite) != target:
        _turn(state, satellite, target, plan)
    state.calibrated.add(instrument)
    plan.append(f"calibrate {satellite} {instrument} {target}")
    return True


def _best_calibration_target(state: SatelliteState, satellite: str, instrument: str) -> Optional[str]:
    targets = state.calibration_targets.get(instrument, [])
    if not targets:
        return None
    current = state.pointing.get(satellite)
    if current in targets:
        return current
    return targets[0]


def _make_power_available(state: SatelliteState, satellite: str, plan: List[str], keep: Optional[str]) -> None:
    if satellite in state.power_avail:
        return
    for instrument in sorted(state.power_on):
        if state.on_board.get(instrument) == satellite and instrument != keep:
            _switch_off(state, instrument, satellite, plan)
            return


def _switch_on(state: SatelliteState, instrument: str, satellite: str, plan: List[str]) -> None:
    state.power_on.add(instrument)
    state.calibrated.discard(instrument)
    state.power_avail.discard(satellite)
    plan.append(f"switch_on {instrument} {satellite}")


def _switch_off(state: SatelliteState, instrument: str, satellite: str, plan: List[str]) -> None:
    state.power_on.discard(instrument)
    state.power_avail.add(satellite)
    plan.append(f"switch_off {instrument} {satellite}")


def _turn(state: SatelliteState, satellite: str, direction: str, plan: List[str]) -> None:
    previous = state.pointing.get(satellite)
    if previous == direction:
        return
    if previous is None:
        previous = direction
    state.pointing[satellite] = direction
    plan.append(f"turn_to {satellite} {direction} {previous}")


def _pointing_conflict(goals: Sequence[Fact]) -> str:
    wanted = {}
    for goal in goals:
        if goal[0] != "pointing":
            continue
        satellite, direction = goal[1], goal[2]
        if satellite in wanted and wanted[satellite] != direction:
            return f"mutually exclusive pointing goals for {satellite}"
        wanted[satellite] = direction
    return ""


def _copy_state(src: SatelliteState, dst: SatelliteState) -> None:
    dst.directions = set(src.directions)
    dst.instruments = set(src.instruments)
    dst.modes = set(src.modes)
    dst.satellites = set(src.satellites)
    dst.on_board = dict(src.on_board)
    dst.supports = {k: set(v) for k, v in src.supports.items()}
    dst.calibration_targets = {k: list(v) for k, v in src.calibration_targets.items()}
    dst.pointing = dict(src.pointing)
    dst.power_avail = set(src.power_avail)
    dst.power_on = set(src.power_on)
    dst.calibrated = set(src.calibrated)
    dst.have_images = set(src.have_images)
