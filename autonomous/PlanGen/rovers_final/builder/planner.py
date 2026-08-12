"""Goal-directed planner for the rovers domain."""

from collections import deque
from typing import Iterable, List, Optional, Set, Tuple

from architect.spec import Fact, PlanResult, RoverProblem, RoverState
from builder.parser import make_state


def construct_plan(problem: RoverProblem) -> PlanResult:
    state = make_state(problem)
    impossible = _static_conflict(problem.goal_facts)
    if impossible:
        return PlanResult(False, reason="mutually exclusive rover/location goals", failed_goal=impossible)

    plan: List[str] = []
    non_position = [g for g in problem.goal_facts if g[0] != "at"]
    position = [g for g in problem.goal_facts if g[0] == "at"]
    for goal in non_position + position:
        if goal_holds(state, goal):
            continue
        before = len(plan)
        if not _achieve_goal(state, goal, plan, protected_goals=problem.goal_facts):
            return PlanResult(False, plan, reason=f"could not achieve {format_fact(goal)}", failed_goal=goal)
        if len(plan) == before and not goal_holds(state, goal):
            return PlanResult(False, plan, reason=f"goal unchanged {format_fact(goal)}", failed_goal=goal)

    if not all(goal_holds(state, g) for g in problem.goal_facts):
        return PlanResult(False, plan, reason="constructed plan does not satisfy all goals")
    return PlanResult(True, plan, reason="constructed deterministic rovers plan")


def goal_holds(state: RoverState, fact: Fact) -> bool:
    name = fact[0]
    if name == "at":
        return state.at.get(fact[1]) == fact[2]
    if name == "at_lander":
        return state.lander_at.get(fact[1]) == fact[2]
    if name == "empty":
        return fact[1] in state.empty
    if name == "full":
        return fact[1] in state.full
    if name == "at_rock_sample":
        return fact[1] in state.rock_samples
    if name == "at_soil_sample":
        return fact[1] in state.soil_samples
    if name == "have_rock_analysis":
        return (fact[1], fact[2]) in state.have_rock
    if name == "have_soil_analysis":
        return (fact[1], fact[2]) in state.have_soil
    if name == "have_image":
        return (fact[1], fact[2], fact[3]) in state.have_image
    if name == "communicated_rock_data":
        return fact[1] in state.communicated_rock
    if name == "communicated_soil_data":
        return fact[1] in state.communicated_soil
    if name == "communicated_image_data":
        return (fact[1], fact[2]) in state.communicated_image
    return fact in state.static


def simulate_plan(problem: RoverProblem, plan: Iterable[str]) -> RoverState:
    state = make_state(problem)
    for step in plan:
        apply_action(state, step.split())
    return state


def goals_hold(state: RoverState, goals: Iterable[Fact]) -> bool:
    return all(goal_holds(state, goal) for goal in goals)


def apply_action(state: RoverState, parts: List[str]) -> None:
    action = parts[0]
    if action == "navigate":
        _, rover, src, dst = parts
        _require(state.at.get(rover) == src and _static(state, "can_traverse", rover, src, dst) and _static(state, "visible", src, dst), parts)
        state.at[rover] = dst
    elif action == "drop":
        _, rover, store = parts
        _require(_static(state, "store_of", store, rover) and store in state.full, parts)
        state.full.discard(store)
        state.empty.add(store)
    elif action == "sample_rock":
        _, rover, store, wp = parts
        _require(state.at.get(rover) == wp and wp in state.rock_samples and _static(state, "equipped_for_rock_analysis", rover), parts)
        _require(_static(state, "store_of", store, rover) and store in state.empty, parts)
        state.empty.discard(store)
        state.full.add(store)
        state.have_rock.add((rover, wp))
        state.rock_samples.discard(wp)
    elif action == "sample_soil":
        _, rover, store, wp = parts
        _require(state.at.get(rover) == wp and wp in state.soil_samples and _static(state, "equipped_for_soil_analysis", rover), parts)
        _require(_static(state, "store_of", store, rover) and store in state.empty, parts)
        state.empty.discard(store)
        state.full.add(store)
        state.have_soil.add((rover, wp))
        state.soil_samples.discard(wp)
    elif action == "calibrate":
        _, rover, camera, target, wp = parts
        _require(state.at.get(rover) == wp and _static(state, "on_board", camera, rover), parts)
        _require(_static(state, "equipped_for_imaging", rover) and _static(state, "calibration_target", camera, target), parts)
        _require(_static(state, "visible_from", target, wp), parts)
        state.calibrated.add((camera, rover))
    elif action == "take_image":
        _, rover, wp, objective, camera, mode = parts
        _require(state.at.get(rover) == wp and (camera, rover) in state.calibrated, parts)
        _require(_static(state, "on_board", camera, rover) and _static(state, "supports", camera, mode), parts)
        _require(_static(state, "visible_from", objective, wp), parts)
        state.have_image.add((rover, objective, mode))
        state.calibrated.discard((camera, rover))
    elif action == "communicate_rock_data":
        _, rover, lander, sample_wp, rover_wp, lander_wp = parts
        _require(state.at.get(rover) == rover_wp and state.lander_at.get(lander) == lander_wp, parts)
        _require((rover, sample_wp) in state.have_rock and _static(state, "visible", rover_wp, lander_wp), parts)
        state.communicated_rock.add(sample_wp)
    elif action == "communicate_soil_data":
        _, rover, lander, sample_wp, rover_wp, lander_wp = parts
        _require(state.at.get(rover) == rover_wp and state.lander_at.get(lander) == lander_wp, parts)
        _require((rover, sample_wp) in state.have_soil and _static(state, "visible", rover_wp, lander_wp), parts)
        state.communicated_soil.add(sample_wp)
    elif action == "communicate_image_data":
        _, rover, lander, objective, mode, rover_wp, lander_wp = parts
        _require(state.at.get(rover) == rover_wp and state.lander_at.get(lander) == lander_wp, parts)
        _require((rover, objective, mode) in state.have_image and _static(state, "visible", rover_wp, lander_wp), parts)
        state.communicated_image.add((objective, mode))
    else:
        raise ValueError(f"Unknown action {action}")


def _achieve_goal(state: RoverState, goal: Fact, plan: List[str], protected_goals: List[Fact]) -> bool:
    name = goal[0]
    if name == "at":
        return _move_to(state, goal[1], goal[2], plan)
    if name == "empty":
        store = goal[1]
        if store in state.empty:
            return True
        rover = _store_rover(state, store)
        if rover and store in state.full:
            _emit(state, plan, f"drop {rover} {store}")
            return True
        return False
    if name == "have_rock_analysis":
        return _ensure_sample_analysis(state, goal[1], goal[2], "rock", plan)
    if name == "have_soil_analysis":
        return _ensure_sample_analysis(state, goal[1], goal[2], "soil", plan)
    if name == "have_image":
        return _ensure_image(state, goal[1], goal[2], goal[3], plan)
    if name == "communicated_rock_data":
        return _ensure_communicated_sample(state, goal[1], "rock", plan)
    if name == "communicated_soil_data":
        return _ensure_communicated_sample(state, goal[1], "soil", plan)
    if name == "communicated_image_data":
        return _ensure_communicated_image(state, goal[1], goal[2], plan)
    if name in {"at_rock_sample", "at_soil_sample", "at_lander", "full"}:
        return goal_holds(state, goal)
    return goal in state.static


def _ensure_sample_analysis(state: RoverState, rover: str, wp: str, kind: str, plan: List[str]) -> bool:
    have = state.have_rock if kind == "rock" else state.have_soil
    sample_set = state.rock_samples if kind == "rock" else state.soil_samples
    equip = "equipped_for_rock_analysis" if kind == "rock" else "equipped_for_soil_analysis"
    action = "sample_rock" if kind == "rock" else "sample_soil"
    if (rover, wp) in have:
        return True
    if wp not in sample_set or not _static(state, equip, rover):
        return False
    store = _store_for_rover(state, rover)
    if not store:
        return False
    if store not in state.empty:
        if store in state.full:
            _emit(state, plan, f"drop {rover} {store}")
        else:
            return False
    if not _move_to(state, rover, wp, plan):
        return False
    _emit(state, plan, f"{action} {rover} {store} {wp}")
    return True


def _ensure_communicated_sample(state: RoverState, wp: str, kind: str, plan: List[str]) -> bool:
    communicated = state.communicated_rock if kind == "rock" else state.communicated_soil
    have = state.have_rock if kind == "rock" else state.have_soil
    equip = "equipped_for_rock_analysis" if kind == "rock" else "equipped_for_soil_analysis"
    if wp in communicated:
        return True
    candidates = [r for r, p in have if p == wp]
    candidates += [r for r in state.rovers if _static(state, equip, r)]
    for rover in _unique(candidates):
        local_plan: List[str] = []
        local_state = state.copy()
        if not _ensure_sample_analysis(local_state, rover, wp, kind, local_plan):
            continue
        comm_wp = _nearest_visible_to_lander(local_state, rover)
        if comm_wp is None or not _move_to(local_state, rover, comm_wp, local_plan):
            continue
        lander, lander_wp = _first_lander(local_state)
        action = "communicate_rock_data" if kind == "rock" else "communicate_soil_data"
        _emit(local_state, local_plan, f"{action} {rover} {lander} {wp} {comm_wp} {lander_wp}")
        _commit(state, plan, local_state, local_plan)
        return True
    return False


def _ensure_image(state: RoverState, rover: str, objective: str, mode: str, plan: List[str]) -> bool:
    if (rover, objective, mode) in state.have_image:
        return True
    if not _static(state, "equipped_for_imaging", rover):
        return False
    for camera in state.cameras:
        if not _static(state, "on_board", camera, rover) or not _static(state, "supports", camera, mode):
            continue
        targets = [f[2] for f in state.static if f[0] == "calibration_target" and f[1] == camera]
        image_wps = _visible_from_wps(state, objective)
        for target in targets:
            calibration_wps = _visible_from_wps(state, target)
            for image_wp in image_wps:
                local_state = state.copy()
                local_plan: List[str] = []
                cal_wp = _best_intermediate(local_state, rover, calibration_wps, image_wp)
                if cal_wp is None:
                    continue
                if not _move_to(local_state, rover, cal_wp, local_plan):
                    continue
                _emit(local_state, local_plan, f"calibrate {rover} {camera} {target} {cal_wp}")
                if not _move_to(local_state, rover, image_wp, local_plan):
                    continue
                _emit(local_state, local_plan, f"take_image {rover} {image_wp} {objective} {camera} {mode}")
                _commit(state, plan, local_state, local_plan)
                return True
    return False


def _ensure_communicated_image(state: RoverState, objective: str, mode: str, plan: List[str]) -> bool:
    if (objective, mode) in state.communicated_image:
        return True
    candidates = [r for r, o, m in state.have_image if o == objective and m == mode]
    candidates += list(state.rovers)
    for rover in _unique(candidates):
        local_state = state.copy()
        local_plan: List[str] = []
        if not _ensure_image(local_state, rover, objective, mode, local_plan):
            continue
        comm_wp = _nearest_visible_to_lander(local_state, rover)
        if comm_wp is None or not _move_to(local_state, rover, comm_wp, local_plan):
            continue
        lander, lander_wp = _first_lander(local_state)
        _emit(local_state, local_plan, f"communicate_image_data {rover} {lander} {objective} {mode} {comm_wp} {lander_wp}")
        _commit(state, plan, local_state, local_plan)
        return True
    return False


def _move_to(state: RoverState, rover: str, dst: str, plan: List[str]) -> bool:
    src = state.at.get(rover)
    if src == dst:
        return True
    path = _shortest_path(state, rover, src, dst)
    if not path:
        return False
    for a, b in zip(path, path[1:]):
        _emit(state, plan, f"navigate {rover} {a} {b}")
    return True


def _shortest_path(state: RoverState, rover: str, src: Optional[str], dst: str) -> Optional[List[str]]:
    if src is None:
        return None
    queue = deque([(src, [src])])
    seen = {src}
    while queue:
        node, path = queue.popleft()
        if node == dst:
            return path
        for nxt in sorted(_neighbors(state, rover, node), key=_name_key):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def _neighbors(state: RoverState, rover: str, wp: str) -> Set[str]:
    return {f[3] for f in state.static if f[0] == "can_traverse" and f[1] == rover and f[2] == wp and _static(state, "visible", wp, f[3])}


def _nearest_visible_to_lander(state: RoverState, rover: str) -> Optional[str]:
    _, lander_wp = _first_lander(state)
    candidates = [wp for wp in state.waypoints if _static(state, "visible", wp, lander_wp)]
    best: Optional[Tuple[int, str]] = None
    for wp in candidates:
        path = _shortest_path(state, rover, state.at.get(rover), wp)
        if path is None:
            continue
        score = (len(path), wp)
        if best is None or score < best:
            best = score
    return best[1] if best else None


def _best_intermediate(state: RoverState, rover: str, candidates: Iterable[str], final_wp: str) -> Optional[str]:
    best: Optional[Tuple[int, str]] = None
    for wp in candidates:
        first = _shortest_path(state, rover, state.at.get(rover), wp)
        if first is None:
            continue
        second = _shortest_path(state, rover, wp, final_wp)
        if second is None:
            continue
        score = (len(first) + len(second), wp)
        if best is None or score < best:
            best = score
    return best[1] if best else None


def _visible_from_wps(state: RoverState, objective: str) -> List[str]:
    return sorted([f[2] for f in state.static if f[0] == "visible_from" and f[1] == objective], key=_name_key)


def _emit(state: RoverState, plan: List[str], action: str) -> None:
    apply_action(state, action.split())
    plan.append(action)


def _commit(state: RoverState, plan: List[str], local_state: RoverState, local_plan: List[str]) -> None:
    state.__dict__.update(local_state.__dict__)
    plan.extend(local_plan)


def _static(state: RoverState, name: str, *args: str) -> bool:
    return (name, *args) in state.static


def _store_for_rover(state: RoverState, rover: str) -> Optional[str]:
    stores = sorted([f[1] for f in state.static if f[0] == "store_of" and f[2] == rover], key=_name_key)
    return stores[0] if stores else None


def _store_rover(state: RoverState, store: str) -> Optional[str]:
    for fact in state.static:
        if fact[0] == "store_of" and fact[1] == store:
            return fact[2]
    return None


def _first_lander(state: RoverState) -> Tuple[str, str]:
    lander = sorted(state.lander_at, key=_name_key)[0]
    return lander, state.lander_at[lander]


def _static_conflict(goals: List[Fact]) -> Optional[Fact]:
    rover_locs = {}
    for goal in goals:
        if goal[0] == "at":
            old = rover_locs.setdefault(goal[1], goal[2])
            if old != goal[2]:
                return goal
    return None


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _require(condition: bool, parts: List[str]) -> None:
    if not condition:
        raise ValueError(f"Invalid action: {' '.join(parts)}")


def _name_key(name: str) -> Tuple[str, int]:
    import re

    match = re.match(r"([A-Za-z_]+)(\d+)$", name)
    if match:
        return (match.group(1), int(match.group(2)))
    return (name, -1)


def format_fact(fact: Fact) -> str:
    return "(" + " ".join(fact) + ")"
