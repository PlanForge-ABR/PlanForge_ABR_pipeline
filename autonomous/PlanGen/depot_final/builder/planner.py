"""Constructive depot planner and action simulator."""

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from architect.spec import DepotGoals, DepotState, Fact, SolveResult


def construct_plan(initial: DepotState, goals: DepotGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)

    state = deepcopy(initial)
    plan: List[str] = []

    try:
        goal_on = [(f[1], f[2]) for f in goals.facts if f[0] == "on"]
        _build_goal_stacks(state, goal_on, plan)
        _satisfy_remaining_goals(state, goals.facts, plan)
    except (RuntimeError, ValueError) as exc:
        return SolveResult(False, [], str(exc))

    try:
        final_state = simulate_plan(initial, plan)
    except Exception as exc:
        return SolveResult(False, [], f"generated plan failed during simulation: {exc}")
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "constructive planner could not satisfy the requested facts")
    return SolveResult(True, plan)


def validate_goal_consistency(state: DepotState, goals: DepotGoals) -> Tuple[bool, str]:
    at_goal: Dict[str, str] = {}
    on_goal: Dict[str, str] = {}
    top_goal: Dict[str, str] = {}
    in_goal: Dict[str, str] = {}
    lifting_goal: Dict[str, str] = {}

    for fact in goals.facts:
        pred = fact[0]
        if pred == "at":
            obj, place = fact[1], fact[2]
            if obj in at_goal and at_goal[obj] != place:
                return False, f"{obj} cannot be at two places"
            at_goal[obj] = place
            if obj in state.hoists + state.pallets and state.at.get(obj) != place:
                return False, f"{obj} has a fixed location and cannot move to {place}"
        elif pred == "on":
            crate, support = fact[1], fact[2]
            if crate == support:
                return False, f"{crate} cannot be on itself"
            if crate in on_goal and on_goal[crate] != support:
                return False, f"{crate} cannot be on two supports"
            if support in top_goal and top_goal[support] != crate:
                return False, f"two crates cannot both be on {support}"
            on_goal[crate] = support
            top_goal[support] = crate
        elif pred == "in":
            crate, truck = fact[1], fact[2]
            if crate in in_goal and in_goal[crate] != truck:
                return False, f"{crate} cannot be in two trucks"
            in_goal[crate] = truck
        elif pred == "lifting":
            hoist, crate = fact[1], fact[2]
            if hoist in lifting_goal and lifting_goal[hoist] != crate:
                return False, f"{hoist} cannot lift two crates"
            lifting_goal[hoist] = crate
        elif pred == "available" and fact[1] in lifting_goal:
            return False, f"{fact[1]} cannot be available and lifting"

    if _has_cycle(on_goal):
        return False, "goal on-relations contain a cycle"

    for crate, support in on_goal.items():
        if crate in in_goal:
            return False, f"{crate} cannot be both on {support} and in a truck"
        if crate in at_goal:
            support_place = state.at.get(support)
            if support_place and at_goal[crate] != support_place:
                return False, f"{crate} cannot be on {support} at a different place"
    for obj, place in at_goal.items():
        if obj in state.crates and obj in state.in_truck:
            continue
        if place not in state.places:
            return False, f"unknown place {place}"
    return True, ""


def _build_goal_stacks(state: DepotState, goal_on: List[Tuple[str, str]], plan: List[str]) -> None:
    support_of = {crate: support for crate, support in goal_on}
    top_of = {support: crate for crate, support in goal_on}
    bases = [support for support in top_of if support not in support_of]
    bases.sort(key=_name_key)

    for base in bases:
        _ensure_surface_clear(state, base, plan)
        parent = base
        while parent in top_of:
            child = top_of[parent]
            dest = _surface_place(state, parent)
            if dest is None:
                raise RuntimeError(f"cannot determine location of support {parent}")
            _move_crate_onto(state, child, parent, dest, plan)
            parent = child


def _satisfy_remaining_goals(state: DepotState, facts: List[Fact], plan: List[str]) -> None:
    for fact in facts:
        pred = fact[0]
        if pred == "at":
            obj, place = fact[1], fact[2]
            if obj in state.trucks:
                _drive(state, plan, obj, place)
            elif obj in state.crates:
                _move_crate_to_place(state, obj, place, plan)
            elif state.at.get(obj) != place:
                raise RuntimeError(f"{obj} cannot move to {place}")
        elif pred == "clear":
            _ensure_clear_fact(state, fact[1], plan)
        elif pred == "available":
            _free_hoist(state, fact[1], plan)
        elif pred == "in":
            _put_crate_in_truck(state, fact[1], fact[2], plan)
        elif pred == "lifting":
            _make_lifting(state, fact[1], fact[2], plan)


def _move_crate_onto(state: DepotState, crate: str, support: str, place: str, plan: List[str]) -> None:
    if state.on.get(crate) == support and state.at.get(crate) == place:
        return
    _ensure_surface_clear(state, support, plan)
    _ensure_crate_mobile(state, crate, plan)
    if _crate_place(state, crate) == place and _is_lifting_crate(state, crate):
        hoist = _hoist_lifting_crate(state, crate)
        _drop(state, plan, hoist, crate, support, place)
        return
    _put_crate_in_truck_at_source(state, crate, plan)
    truck = state.in_truck[crate]
    _drive(state, plan, truck, place)
    hoist = _available_hoist_at(state, place, plan)
    _unload(state, plan, hoist, crate, truck, place)
    _drop(state, plan, hoist, crate, support, place)


def _move_crate_to_place(state: DepotState, crate: str, place: str, plan: List[str]) -> None:
    if state.at.get(crate) == place:
        return
    _ensure_crate_mobile(state, crate, plan)
    _put_crate_in_truck_at_source(state, crate, plan)
    truck = state.in_truck[crate]
    _drive(state, plan, truck, place)
    support = _find_clear_surface_at(state, place, avoid={crate})
    if support is None:
        raise RuntimeError(f"no clear surface at {place} for {crate}")
    hoist = _available_hoist_at(state, place, plan)
    _unload(state, plan, hoist, crate, truck, place)
    _drop(state, plan, hoist, crate, support, place)


def _put_crate_in_truck(state: DepotState, crate: str, truck: str, plan: List[str]) -> None:
    if state.in_truck.get(crate) == truck:
        return
    _ensure_crate_mobile(state, crate, plan)
    if _is_lifting_crate(state, crate):
        place = _crate_place(state, crate)
        _drive(state, plan, truck, place)
        _load(state, plan, _hoist_lifting_crate(state, crate), crate, truck, place)
    else:
        _put_crate_in_truck_at_source(state, crate, plan, preferred_truck=truck)
        if state.in_truck.get(crate) != truck:
            raise RuntimeError(f"could not load {crate} into {truck}")


def _make_lifting(state: DepotState, hoist: str, crate: str, plan: List[str]) -> None:
    if state.lifting.get(hoist) == crate:
        return
    _free_hoist(state, hoist, plan)
    place = state.at[hoist]
    _move_crate_to_place(state, crate, place, plan)
    _ensure_crate_mobile(state, crate, plan)
    if state.in_truck.get(crate):
        truck = state.in_truck[crate]
        _drive(state, plan, truck, place)
        _unload(state, plan, hoist, crate, truck, place)
    elif state.at.get(crate) == place and crate in state.on:
        _lift(state, plan, hoist, crate, state.on[crate], place)
    else:
        raise RuntimeError(f"could not make {hoist} lift {crate}")


def _ensure_clear_fact(state: DepotState, surface: str, plan: List[str]) -> None:
    if surface in state.clear:
        return
    if surface in state.crates and _is_lifting_crate(state, surface):
        place = _crate_place(state, surface)
        support = _find_clear_surface_at(state, place, avoid={surface})
        if support:
            _drop(state, plan, _hoist_lifting_crate(state, surface), surface, support, place)
            return
    _ensure_surface_clear(state, surface, plan)


def _ensure_surface_clear(state: DepotState, surface: str, plan: List[str]) -> None:
    while surface not in state.clear:
        top = _top_crate_on(state, surface)
        if top is None:
            raise RuntimeError(f"{surface} cannot be made clear")
        _put_crate_in_truck_at_source(state, top, plan)


def _ensure_crate_mobile(state: DepotState, crate: str, plan: List[str]) -> None:
    if crate in state.in_truck or _is_lifting_crate(state, crate):
        return
    _ensure_surface_clear(state, crate, plan)


def _put_crate_in_truck_at_source(
    state: DepotState, crate: str, plan: List[str], preferred_truck: Optional[str] = None
) -> None:
    if crate in state.in_truck:
        return
    _ensure_surface_clear(state, crate, plan)
    if _is_lifting_crate(state, crate):
        place = _crate_place(state, crate)
        truck = preferred_truck or _nearest_truck(state, place)
        _drive(state, plan, truck, place)
        _load(state, plan, _hoist_lifting_crate(state, crate), crate, truck, place)
        return
    if crate not in state.on:
        raise RuntimeError(f"{crate} is not on a liftable surface")
    place = state.at.get(crate)
    if place is None:
        raise RuntimeError(f"{crate} has no known source place")
    hoist = _available_hoist_at(state, place, plan)
    _lift(state, plan, hoist, crate, state.on[crate], place)
    truck = preferred_truck or _nearest_truck(state, place)
    _drive(state, plan, truck, place)
    _load(state, plan, hoist, crate, truck, place)


def _free_hoist(state: DepotState, hoist: str, plan: List[str]) -> None:
    if hoist in state.available:
        return
    crate = state.lifting.get(hoist)
    if not crate:
        raise RuntimeError(f"{hoist} is neither available nor lifting a crate")
    place = state.at[hoist]
    support = _find_clear_surface_at(state, place, avoid={crate})
    if support:
        _drop(state, plan, hoist, crate, support, place)
    else:
        truck = _nearest_truck(state, place)
        _drive(state, plan, truck, place)
        _load(state, plan, hoist, crate, truck, place)


def _available_hoist_at(state: DepotState, place: str, plan: List[str]) -> str:
    hoists = [h for h in state.hoists if state.at.get(h) == place]
    if not hoists:
        raise RuntimeError(f"no hoist exists at {place}")
    hoist = sorted(hoists, key=_name_key)[0]
    _free_hoist(state, hoist, plan)
    return hoist


def _nearest_truck(state: DepotState, place: str) -> str:
    if not state.trucks:
        raise RuntimeError("no trucks are available")
    at_place = [t for t in state.trucks if state.at.get(t) == place]
    return sorted(at_place or state.trucks, key=_name_key)[0]


def _find_clear_surface_at(state: DepotState, place: str, avoid: Set[str]) -> Optional[str]:
    surfaces = state.pallets + state.crates
    for surface in sorted(surfaces, key=_name_key):
        if surface in avoid:
            continue
        if surface in state.clear and state.at.get(surface) == place:
            return surface
    return None


def _surface_place(state: DepotState, surface: str) -> Optional[str]:
    return state.at.get(surface)


def _crate_place(state: DepotState, crate: str) -> Optional[str]:
    if crate in state.at:
        return state.at[crate]
    if crate in state.in_truck:
        return state.at.get(state.in_truck[crate])
    hoist = _hoist_lifting_crate(state, crate)
    if hoist:
        return state.at.get(hoist)
    return None


def _top_crate_on(state: DepotState, surface: str) -> Optional[str]:
    for crate, support in sorted(state.on.items(), key=lambda item: _name_key(item[0])):
        if support == surface:
            return crate
    return None


def _is_lifting_crate(state: DepotState, crate: str) -> bool:
    return _hoist_lifting_crate(state, crate) is not None


def _hoist_lifting_crate(state: DepotState, crate: str) -> Optional[str]:
    for hoist, lifted in state.lifting.items():
        if lifted == crate:
            return hoist
    return None


def _drive(state: DepotState, plan: Optional[List[str]], truck: str, dest: str) -> None:
    src = state.at.get(truck)
    if src == dest:
        return
    if src is None:
        raise ValueError(f"{truck} has no current location")
    state.at[truck] = dest
    if plan is not None:
        plan.append(f"drive {truck} {src} {dest}")


def _lift(state: DepotState, plan: Optional[List[str]], hoist: str, crate: str, support: str, place: str) -> None:
    if state.at.get(hoist) != place or hoist not in state.available:
        raise ValueError(f"{hoist} is not available at {place}")
    if state.at.get(crate) != place or state.on.get(crate) != support or crate not in state.clear:
        raise ValueError(f"{crate} cannot be lifted from {support} at {place}")
    state.at.pop(crate, None)
    state.on.pop(crate, None)
    state.clear.discard(crate)
    state.clear.add(support)
    state.available.discard(hoist)
    state.lifting[hoist] = crate
    if plan is not None:
        plan.append(f"lift {hoist} {crate} {support} {place}")


def _drop(state: DepotState, plan: Optional[List[str]], hoist: str, crate: str, support: str, place: str) -> None:
    if state.at.get(hoist) != place or state.at.get(support) != place:
        raise ValueError(f"{hoist} and {support} are not both at {place}")
    if state.lifting.get(hoist) != crate or support not in state.clear:
        raise ValueError(f"{crate} cannot be dropped on {support}")
    state.lifting.pop(hoist)
    state.available.add(hoist)
    state.at[crate] = place
    state.on[crate] = support
    state.clear.discard(support)
    state.clear.add(crate)
    if plan is not None:
        plan.append(f"drop {hoist} {crate} {support} {place}")


def _load(state: DepotState, plan: Optional[List[str]], hoist: str, crate: str, truck: str, place: str) -> None:
    if state.at.get(hoist) != place or state.at.get(truck) != place or state.lifting.get(hoist) != crate:
        raise ValueError(f"{crate} cannot be loaded into {truck} at {place}")
    state.lifting.pop(hoist)
    state.available.add(hoist)
    state.in_truck[crate] = truck
    if plan is not None:
        plan.append(f"load {hoist} {crate} {truck} {place}")


def _unload(state: DepotState, plan: Optional[List[str]], hoist: str, crate: str, truck: str, place: str) -> None:
    if state.at.get(hoist) != place or state.at.get(truck) != place:
        raise ValueError(f"{hoist} and {truck} are not both at {place}")
    if hoist not in state.available or state.in_truck.get(crate) != truck:
        raise ValueError(f"{crate} cannot be unloaded from {truck}")
    state.in_truck.pop(crate)
    state.available.discard(hoist)
    state.lifting[hoist] = crate
    if plan is not None:
        plan.append(f"unload {hoist} {crate} {truck} {place}")


def simulate_plan(initial: DepotState, plan: List[str]) -> DepotState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        name = parts[0]
        if name == "drive":
            _drive(state, None, parts[1], parts[3])
        elif name == "lift":
            _lift(state, None, parts[1], parts[2], parts[3], parts[4])
        elif name == "drop":
            _drop(state, None, parts[1], parts[2], parts[3], parts[4])
        elif name == "load":
            _load(state, None, parts[1], parts[2], parts[3], parts[4])
        elif name == "unload":
            _unload(state, None, parts[1], parts[2], parts[3], parts[4])
        else:
            raise ValueError(f"unknown action {action}")
    return state


def goals_hold(state: DepotState, goals: DepotGoals) -> bool:
    for fact in goals.facts:
        pred = fact[0]
        if pred == "at" and state.at.get(fact[1]) != fact[2]:
            return False
        if pred == "available" and fact[1] not in state.available:
            return False
        if pred == "clear" and fact[1] not in state.clear:
            return False
        if pred == "on" and state.on.get(fact[1]) != fact[2]:
            return False
        if pred == "in" and state.in_truck.get(fact[1]) != fact[2]:
            return False
        if pred == "lifting" and state.lifting.get(fact[1]) != fact[2]:
            return False
    return True


def _has_cycle(support_of: Dict[str, str]) -> bool:
    for start in support_of:
        seen = set()
        cur = start
        while cur in support_of:
            if cur in seen:
                return True
            seen.add(cur)
            cur = support_of[cur]
    return False


def _name_key(name: str):
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return (prefix, int(digits) if digits else -1, name)
