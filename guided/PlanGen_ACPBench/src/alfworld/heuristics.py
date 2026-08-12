from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

Fact = Tuple[str, Tuple[str, ...]]

VALIDATION_STEP_COST = 1.0
PROPERTY_FIX_COST = 2.0
LARGE_PENALTY = 100.0


def _extract_state(obj: Any) -> Sequence[Any]:
    if isinstance(obj, dict):
        state = obj.get("state", [])
        if isinstance(state, (list, tuple)):
            return state
        return []
    if isinstance(obj, (list, tuple)):
        return obj
    return []


def _as_fact(entry: Any) -> Fact:
    if isinstance(entry, tuple):
        return entry
    pred = str(entry.get("predicate"))
    args = entry.get("args", [])
    if not isinstance(args, (list, tuple)):
        args = [args]
    return pred, tuple(str(a) for a in args)


class StateSummary:
    """Thin index over the state to make domain-specific checks cheap."""

    def __init__(self, entries: Sequence[Any]):
        self.facts: Set[Fact] = set()

        self.object_types: Dict[str, Set[str]] = defaultdict(set)
        self.receptacle_types: Dict[str, Set[str]] = defaultdict(set)
        self.objects_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.receptacles_by_type: Dict[str, Set[str]] = defaultdict(set)

        self.in_receptacle: Dict[str, Set[str]] = defaultdict(set)
        self.receptacle_contents: Dict[str, Set[str]] = defaultdict(set)

        self.object_locations: Dict[str, Set[str]] = defaultdict(set)
        self.receptacle_locations: Dict[str, str] = {}
        self.agent_locations: Dict[str, str] = {}

        self.holds: Dict[str, Set[str]] = defaultdict(set)
        self.object_held_by: Dict[str, Set[str]] = defaultdict(set)

        self.cleanable: Set[str] = set()
        self.coolable: Set[str] = set()
        self.heatable: Set[str] = set()

        self.isclean: Set[str] = set()
        self.iscool: Set[str] = set()
        self.ishot: Set[str] = set()

        self.toggleable: Set[str] = set()
        self.istoggled: Set[str] = set()

        self.open: Set[str] = set()
        self.closed: Set[str] = set()

        for entry in entries:
            fact = _as_fact(entry)
            self.facts.add(fact)

            pred, args = fact
            if pred == "objecttype" and len(args) == 2:
                obj, typ = args
                self.object_types[obj].add(typ)
                self.objects_by_type[typ].add(obj)
            elif pred == "receptacletype" and len(args) == 2:
                rec, rtype = args
                self.receptacle_types[rec].add(rtype)
                self.receptacles_by_type[rtype].add(rec)
            elif pred == "inreceptacle" and len(args) == 2:
                obj, rec = args
                self.in_receptacle[obj].add(rec)
                self.receptacle_contents[rec].add(obj)
            elif pred == "receptacleatlocation" and len(args) == 2:
                rec, loc = args
                self.receptacle_locations[rec] = loc
            elif pred == "objectatlocation" and len(args) == 2:
                obj, loc = args
                self.object_locations[obj].add(loc)
            elif pred == "atlocation" and len(args) == 2:
                agent, loc = args
                self.agent_locations[agent] = loc
            elif pred == "holds" and len(args) == 2:
                agent, obj = args
                self.holds[agent].add(obj)
                self.object_held_by[obj].add(agent)
            elif pred == "cleanable" and len(args) == 1:
                self.cleanable.add(args[0])
            elif pred == "coolable" and len(args) == 1:
                self.coolable.add(args[0])
            elif pred == "heatable" and len(args) == 1:
                self.heatable.add(args[0])
            elif pred == "isclean" and len(args) == 1:
                self.isclean.add(args[0])
            elif pred == "iscool" and len(args) == 1:
                self.iscool.add(args[0])
            elif pred == "ishot" and len(args) == 1:
                self.ishot.add(args[0])
            elif pred == "toggleable" and len(args) == 1:
                self.toggleable.add(args[0])
            elif pred == "istoggled" and len(args) == 1:
                self.istoggled.add(args[0])
            elif pred == "open" and len(args) == 1:
                self.open.add(args[0])
            elif pred == "closed" and len(args) == 1:
                self.closed.add(args[0])

        self.toggleable_types: Set[str] = set()
        self.toggled_types: Set[str] = set()
        self.held_types: Set[str] = set()

        for obj, types in self.object_types.items():
            if obj in self.toggleable:
                self.toggleable_types.update(types)
            if obj in self.istoggled:
                self.toggled_types.update(types)
            if obj in self.object_held_by:
                self.held_types.update(types)

    def objects_of_type(self, obj_type: str) -> Set[str]:
        return self.objects_by_type.get(obj_type, set())

    def receptacles_of_type(self, rec_type: str) -> Set[str]:
        return self.receptacles_by_type.get(rec_type, set())

    def object_in_receptacle_type(self, obj: str, rec_type: str) -> bool:
        for rec in self.in_receptacle.get(obj, set()):
            if rec_type in self.receptacle_types.get(rec, set()):
                return True
        return False

    def target_locations(self, rec_type: str) -> Set[str]:
        locs: Set[str] = set()
        for rec in self.receptacles_of_type(rec_type):
            loc = self.receptacle_locations.get(rec)
            if loc:
                locs.add(loc)
        return locs

    def held_locations(self, obj: str) -> Set[str]:
        locs: Set[str] = set()
        for agent in self.object_held_by.get(obj, set()):
            loc = self.agent_locations.get(agent)
            if loc:
                locs.add(loc)
        return locs


def _property_cost(summary: StateSummary, obj: str, requirement: str) -> float:
    if requirement == "clean":
        if obj not in summary.cleanable:
            return LARGE_PENALTY
        return 0.0 if obj in summary.isclean else PROPERTY_FIX_COST
    if requirement == "cool":
        if obj not in summary.coolable:
            return LARGE_PENALTY
        return 0.0 if obj in summary.iscool else PROPERTY_FIX_COST
    if requirement == "heat":
        if obj not in summary.heatable:
            return LARGE_PENALTY
        return 0.0 if obj in summary.ishot else PROPERTY_FIX_COST
    return 0.0


def _placement_cost(summary: StateSummary, obj: str, rec_type: str) -> float:
    target_locs = summary.target_locations(rec_type)

    if summary.object_in_receptacle_type(obj, rec_type):
        return 0.0

    held_locs = summary.held_locations(obj)
    if held_locs:
        if target_locs and held_locs & target_locs:
            return 1.0
        return 2.0

    obj_locs = summary.object_locations.get(obj, set())

    base_cost = 4.0
    if target_locs and obj_locs & target_locs:
        base_cost = 2.0
    elif summary.in_receptacle.get(obj):
        # Check if any receptacle containing it is closed
        is_closed = False
        recs = summary.in_receptacle[obj]
        for r in recs:
             if r in summary.closed:
                 is_closed = True
                 break
        if is_closed:
             base_cost = 3.1
        else:
             base_cost = 3.0
    elif obj_locs:
        base_cost = 3.5
    else:
        return 4.0

    # If agent is at the object's location, reduce cost to guide navigation
    agent_locs = set(summary.agent_locations.values())
    


    if obj_locs & agent_locs:
        return base_cost - 0.2
    
    return base_cost


def _score_validate_place(
    summary: StateSummary, obj_type: str, rec_type: str, requirements: Iterable[str] = ()
) -> float:
    objects = summary.objects_of_type(obj_type)
    receptacles = summary.receptacles_of_type(rec_type)
    if not objects or not receptacles:
        return LARGE_PENALTY

    best = LARGE_PENALTY
    for obj in objects:
        prop_cost = sum(_property_cost(summary, obj, req) for req in requirements)
        if prop_cost >= LARGE_PENALTY:
            continue
        placement_cost = _placement_cost(summary, obj, rec_type)
        best = min(best, prop_cost + placement_cost)

    if best >= LARGE_PENALTY:
        return LARGE_PENALTY
    return best + VALIDATION_STEP_COST


def _score_validate_pick_two(summary: StateSummary, obj_type: str, rec_type: str) -> float:
    objects = summary.objects_of_type(obj_type)
    target_recs = summary.receptacles_of_type(rec_type)
    if len(objects) < 2 or not target_recs:
        return LARGE_PENALTY

    # Already satisfied when two are in any receptacle of the right type.
    already = [
        obj for obj in objects if summary.object_in_receptacle_type(obj, rec_type)
    ]
    if len(already) >= 2:
        return VALIDATION_STEP_COST

    best = LARGE_PENALTY
    for rec in target_recs:
        contents = summary.receptacle_contents.get(rec, set())
        count = len([obj for obj in contents if obj in objects])
        missing = max(0, 2 - count)
        if missing == 0:
            return VALIDATION_STEP_COST

        remaining = [obj for obj in objects if obj not in contents]
        placement_scores = sorted(_placement_cost(summary, obj, rec_type) for obj in remaining)
        if len(placement_scores) >= missing:
            cost = sum(placement_scores[:missing]) + VALIDATION_STEP_COST + missing
            best = min(best, cost)

    if best < LARGE_PENALTY:
        return best

    placement_scores = sorted(_placement_cost(summary, obj, rec_type) for obj in objects)
    if len(placement_scores) < 2:
        return LARGE_PENALTY
    return sum(placement_scores[:2]) + 2.0 + VALIDATION_STEP_COST


def _score_validate_examine(summary: StateSummary, toggle_type: str, held_type: str) -> float:
    if toggle_type in summary.toggled_types and held_type in summary.held_types:
        return VALIDATION_STEP_COST

    if toggle_type not in summary.toggleable_types:
        return LARGE_PENALTY

    toggle_cost = 0.0 if toggle_type in summary.toggled_types else PROPERTY_FIX_COST
    if held_type in summary.held_types:
        hold_cost = 0.0
    elif summary.objects_of_type(held_type):
        hold_cost = PROPERTY_FIX_COST
    else:
        return LARGE_PENALTY

    return VALIDATION_STEP_COST + toggle_cost + hold_cost


def _open_cost(summary: StateSummary, obj: str) -> float:
    if obj in summary.open:
        return 0.0

    # Receptacle location
    rec_loc = summary.receptacle_locations.get(obj)
    if not rec_loc:
        # Maybe it's an object acting as receptacle or location logic is different?
        # Fallback to object locations if not in receptacle_locations
        obj_locs = summary.object_locations.get(obj, set())
        if obj_locs:
             rec_loc = list(obj_locs)[0]
    
    if not rec_loc:
        return 2.0 # Assume move + open

    agent_locs = set(summary.agent_locations.values())
    if rec_loc in agent_locs:
        return 1.0 # At location, just Action
    
    # Needs move (1.0) + Action (1.0)
    return 2.0


def heuristic(state: Any, goal: Any) -> float:
    """Domain-aware heuristic for ALFWorld / ALFRED."""
    state_entries = _extract_state(state)
    goal_entries = _extract_state(goal)

    summary = StateSummary(state_entries)
    state_facts = summary.facts
    goal_facts = {_as_fact(entry) for entry in goal_entries}

    total = 0.0
    for fact in goal_facts:
        if fact in state_facts:
            continue
        pred, args = fact
        if pred == "validatepickandplace" and len(args) == 2:
            total += _score_validate_place(summary, args[0], args[1])
        elif pred == "validatecleanandplace" and len(args) == 2:
            total += _score_validate_place(summary, args[0], args[1], ["clean"])
        elif pred == "validatecoolandplace" and len(args) == 2:
            total += _score_validate_place(summary, args[0], args[1], ["cool"])
        elif pred == "validateheatandplace" and len(args) == 2:
            total += _score_validate_place(summary, args[0], args[1], ["heat"])
        elif pred == "validatepicktwoandplace" and len(args) == 2:
            total += _score_validate_pick_two(summary, args[0], args[1])
        elif pred == "validateexamineinlight" and len(args) == 2:
            total += _score_validate_examine(summary, args[0], args[1])
        elif pred == "open" and len(args) == 1:
            total += _open_cost(summary, args[0])
        elif pred == "atlocation" and len(args) == 2:
             # Agent at location?
             agent, target_loc = args
             curr_loc = summary.agent_locations.get(agent)
             if curr_loc == target_loc:
                 total += 0.0
             else:
                 total += 1.0 
        elif pred == "notvalidated":
            # Once removed, we cannot directly reintroduce it; penalize heavily.
            total += LARGE_PENALTY / 2
        else:
            total += 1.0

    return float(total)
