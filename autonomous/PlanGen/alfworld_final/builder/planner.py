"""Builder implementation of the architect's ALFWorld methods."""

from copy import deepcopy
from typing import Dict, List, Optional, Set

from architect.spec import AWGoals, AWState, DirectGoal, SolveResult, ValidationGoal


PROCESS_RECEPTACLE = {
    "clean": ("sinkbasintype", "clean_object"),
    "hot": ("microwavetype", "heat_object"),
    "cool": ("fridgetype", "cool_object"),
}


def construct_plan(initial: AWState, goals: AWGoals) -> SolveResult:
    state = deepcopy(initial)
    plan: List[str] = []

    ok, reason = _check_direct_feasibility(state, goals.direct)
    if not ok:
        return SolveResult(False, [], reason)

    for goal in goals.validations:
        if goal.key in state.validated:
            continue
        ok, reason = _plan_validation(state, goal, plan)
        if not ok:
            return SolveResult(False, [], reason)

    for direct in goals.direct:
        ok, reason = _satisfy_direct(state, direct, plan)
        if not ok:
            return SolveResult(False, [], reason)

    final_state = simulate_plan(initial, plan)
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "constructed plan did not satisfy all parsed goals")
    return SolveResult(True, plan)


def _check_direct_feasibility(state: AWState, direct_goals: List[DirectGoal]):
    agent_locations = {g.value for g in direct_goals if g.kind == "agent_at"}
    if len(agent_locations) > 1:
        return False, "agent cannot be at multiple locations simultaneously"
    for goal in direct_goals:
        if goal.kind == "open" and goal.subject not in state.closed_receptacles and goal.subject not in state.open_receptacles:
            return False, f"{goal.subject} is not openable"
        if goal.kind == "toggled" and goal.subject not in state.toggled_objects and _name_to_type(goal.subject) not in {"desklamptype", "lightswitchtype"}:
            return False, f"{goal.subject} is not a toggleable object"
    return True, ""


def _plan_validation(state: AWState, goal: ValidationGoal, plan: List[str]):
    if goal.property_name == "examined":
        return _plan_examine(state, goal, plan)

    objects = _available_objects(state, goal.object_type)
    if len(objects) < goal.count:
        return False, f"not enough objects of type {goal.object_type}"

    receptacles = _candidate_receptacles(state, goal.receptacle_type)
    if not receptacles:
        return False, f"no receptacle of type {goal.receptacle_type}"

    for obj in objects[: goal.count]:
        target = _choose_target_receptacle(state, receptacles, obj)
        if goal.property_name in PROCESS_RECEPTACLE:
            ok, reason = _process_object(state, obj, goal.property_name, plan)
            if not ok:
                return ok, reason
        if state.object_receptacle.get(obj) != target:
            _move_object_to_receptacle(state, obj, target, plan)

    verb = {
        None: "validate_pick_and_place_in_receptacle",
        "clean": "validate_clean_and_place_in_receptacle",
        "hot": "validate_heat_and_place_in_receptacle",
        "cool": "validate_cool_and_place_in_receptacle",
    }[goal.property_name]
    for obj in objects[: goal.count]:
        target = state.object_receptacle.get(obj) or _choose_target_receptacle(state, receptacles, obj)
        plan.append(f"{verb} {obj} {goal.object_type} {target} {goal.receptacle_type}")
    state.validated.add(goal.key)
    return True, ""


def _plan_examine(state: AWState, goal: ValidationGoal, plan: List[str]):
    objects = _available_objects(state, goal.object_type)
    lamps = _available_objects(state, goal.tool_type or "desklamptype")
    if len(objects) < goal.count or not lamps:
        return False, "missing object or lamp for examine goal"
    lamp = lamps[0]
    lamp_rec = state.object_receptacle.get(lamp)
    if not lamp_rec:
        return False, f"{lamp} has no known receptacle"
    lamp_loc = state.receptacle_locations.get(lamp_rec, state.object_locations.get(lamp, ""))
    for obj in objects[: goal.count]:
        if state.holding != obj:
            _pickup_object(state, obj, plan)
        _go_to(state, lamp_loc, lamp_rec, plan)
        if lamp not in state.toggled_objects:
            plan.append(f"toggle_object_on agent1 {lamp_loc} {lamp} {lamp_rec}")
            state.toggled_objects.add(lamp)
        plan.append(f"validate_examine_in_light {lamp} {goal.tool_type or _name_to_type(lamp)} {obj} {goal.object_type} {lamp_rec} agent1 {lamp_loc}")
    state.validated.add(goal.key)
    return True, ""


def _process_object(state: AWState, obj: str, prop: str, plan: List[str]):
    processor_type, action = PROCESS_RECEPTACLE[prop]
    processors = _candidate_receptacles(state, processor_type)
    if not processors:
        return False, f"no processor receptacle of type {processor_type}"
    processor = processors[0]
    loc = state.receptacle_locations[processor]
    if state.holding != obj:
        _pickup_object(state, obj, plan)
    _go_to(state, loc, processor, plan)
    if processor in state.closed_receptacles:
        _open_receptacle(state, processor, plan)
    plan.append(f"{action} agent1 {loc} {processor} {obj}")
    return True, ""


def _move_object_to_receptacle(state: AWState, obj: str, rec: str, plan: List[str]):
    if state.holding != obj:
        _pickup_object(state, obj, plan)
    loc = state.receptacle_locations[rec]
    _go_to(state, loc, rec, plan)
    if rec in state.closed_receptacles:
        _open_receptacle(state, rec, plan)
    action = "put_object_in_openable_receptacle" if _is_openable(state, rec) else "put_object_on_not_openable_receptacle"
    plan.append(f"{action} agent1 {loc} {obj} {rec} {_name_to_type(obj)} {_name_to_type(rec)}")
    state.object_receptacle[obj] = rec
    state.object_locations[obj] = loc
    state.holding = None


def _pickup_object(state: AWState, obj: str, plan: List[str]):
    if state.holding == obj:
        return
    if state.holding:
        return
    rec = state.object_receptacle.get(obj)
    loc = state.object_locations.get(obj)
    if rec:
        loc = state.receptacle_locations.get(rec, loc)
    if not loc:
        raise ValueError(f"unknown object location for {obj}")
    _go_to(state, loc, rec, plan)
    if rec and rec in state.closed_receptacles:
        _open_receptacle(state, rec, plan)
    if rec and _is_openable(state, rec):
        plan.append(f"pickup_object_from_openable_receptacle agent1 {loc} {obj} {rec}")
    elif rec:
        plan.append(f"pickup_object_from_not_openable_receptacle agent1 {loc} {obj} {rec}")
    else:
        plan.append(f"pickup_object agent1 {loc} {obj}")
    state.holding = obj
    state.object_receptacle.pop(obj, None)


def _satisfy_direct(state: AWState, goal: DirectGoal, plan: List[str]):
    if goal.kind == "agent_at":
        _go_to(state, goal.value or "", None, plan)
        return True, ""
    if goal.kind == "open":
        if goal.subject not in state.open_receptacles:
            _open_receptacle(state, goal.subject, plan)
        return True, ""
    if goal.kind == "toggled":
        if goal.subject not in state.toggled_objects:
            loc = state.object_locations.get(goal.subject)
            if not loc:
                return False, f"unknown toggle target {goal.subject}"
            _go_to(state, loc, None, plan)
            plan.append(f"toggle_object_on agent1 {loc} {goal.subject}")
            state.toggled_objects.add(goal.subject)
        return True, ""
    if goal.kind == "object_in_receptacle":
        if goal.subject not in state.object_locations or goal.value not in state.receptacle_locations:
            return False, "unknown object or receptacle"
        if state.object_receptacle.get(goal.subject) != goal.value:
            _move_object_to_receptacle(state, goal.subject, goal.value or "", plan)
        return True, ""
    return False, f"unsupported direct goal {goal.kind}"


def _open_receptacle(state: AWState, rec: str, plan: List[str]):
    loc = state.receptacle_locations[rec]
    _go_to(state, loc, rec, plan)
    plan.append(f"open_receptacle agent1 {loc} {rec}")
    state.closed_receptacles.discard(rec)
    state.open_receptacles.add(rec)


def _go_to(state: AWState, target: str, focus: Optional[str], plan: List[str]):
    if not target or state.agent_location == target:
        return
    focus_arg = focus or _focus_at_location(state, target) or target
    plan.append(f"go_to_location agent1 {state.agent_location} {target} {focus_arg}")
    state.agent_location = target


def _available_objects(state: AWState, object_type: str) -> List[str]:
    objects = state.objects_by_type.get(object_type, [])
    return [obj for obj in objects if obj in state.object_locations or obj == state.holding]


def _candidate_receptacles(state: AWState, receptacle_type: Optional[str]) -> List[str]:
    if not receptacle_type:
        return []
    return [rec for rec in state.receptacles_by_type.get(receptacle_type, []) if rec in state.receptacle_locations]


def _choose_target_receptacle(state: AWState, receptacles: List[str], obj: str) -> str:
    current = state.object_receptacle.get(obj)
    if current in receptacles:
        return current
    scored = sorted(receptacles, key=lambda r: (r in state.closed_receptacles, _natural_key(r)))
    return scored[0]


def _is_openable(state: AWState, rec: str) -> bool:
    return rec in state.open_receptacles or rec in state.closed_receptacles


def _focus_at_location(state: AWState, loc: str):
    for rec, rec_loc in sorted(state.receptacle_locations.items(), key=lambda kv: _natural_key(kv[0])):
        if rec_loc == loc:
            return rec
    return None


def simulate_plan(initial: AWState, plan: List[str]) -> AWState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        name = parts[0]
        if name == "go_to_location":
            state.agent_location = parts[3]
        elif name == "open_receptacle":
            state.closed_receptacles.discard(parts[3])
            state.open_receptacles.add(parts[3])
        elif name.startswith("pickup_object"):
            obj = parts[3]
            state.holding = obj
            state.object_receptacle.pop(obj, None)
        elif name.startswith("put_object"):
            obj, rec = parts[3], parts[4]
            state.holding = None
            state.object_receptacle[obj] = rec
            state.object_locations[obj] = state.receptacle_locations.get(rec, state.agent_location)
        elif name == "toggle_object_on":
            state.toggled_objects.add(parts[3])
        elif name.startswith("validate_"):
            state.validated.add(action)
    return state


def goals_hold(state: AWState, goals: AWGoals) -> bool:
    agent_locations = [g.value for g in goals.direct if g.kind == "agent_at"]
    if len(set(agent_locations)) > 1:
        return False
    for goal in goals.direct:
        if goal.kind == "agent_at" and state.agent_location != goal.value:
            return False
        if goal.kind == "open" and goal.subject not in state.open_receptacles:
            return False
        if goal.kind == "toggled" and goal.subject not in state.toggled_objects:
            return False
        if goal.kind == "object_in_receptacle" and state.object_receptacle.get(goal.subject) != goal.value:
            return False
    return True


def _name_to_type(name: str) -> str:
    return "".join(ch for ch in name if not ch.isdigit()) + "type"


def _natural_key(name: str):
    head = "".join(ch for ch in name if not ch.isdigit())
    num = "".join(ch for ch in name if ch.isdigit())
    return (head, int(num or 0))
