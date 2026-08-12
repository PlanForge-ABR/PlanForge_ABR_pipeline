from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Set, Tuple

Fact = Tuple[str, Tuple[str, ...]]


def _normalize_facts(items: Iterable[Any]) -> Set[Fact]:
    facts: Set[Fact] = set()
    # Optimization: if input is already a tuple of facts
    if isinstance(items, tuple):
        return set(items)
        
    for entry in items or []:
        # Handle mix of dicts and tuples if necessary
        if isinstance(entry, tuple):
            facts.add(entry)
            continue
            
        pred = str(entry.get("predicate"))
        args = entry.get("args", [])
        if isinstance(args, str):
            args = [args]
        facts.add((pred, tuple(str(a) for a in args)))
    return facts


def _denormalize_facts(facts: Set[Fact]) -> Tuple[Fact, ...]:
    ordered = sorted(facts, key=lambda item: (item[0], item[1]))
    # Return tuple of facts (optimized)
    return tuple(ordered)


def _fact(pred: str, *args: str) -> Fact:
    return (pred, tuple(args))


def _split_name(name: str) -> Tuple[str, int]:
    """Split an identifier into (base, numeric_suffix)."""
    i = len(name)
    while i > 0 and name[i - 1].isdigit():
        i -= 1
    base = name[:i]
    if i == len(name):
        return base, 0
    try:
        return base, int(name[i:])
    except ValueError:
        return base, 0


def _apply(base: Set[Fact], adds: Iterable[Fact] = (), removes: Iterable[Fact] = ()) -> Set[Fact]:
    new_facts = set(base)
    for fact in removes:
        new_facts.discard(fact)
    for fact in adds:
        new_facts.add(fact)
    return new_facts


class IndexedState:
    def __init__(self, facts: Set[Fact]):
        self.facts = facts
        self.agent_locations: Dict[str, str] = {}
        self.receptacle_locations: Dict[str, str] = {}
        self.receptacles_by_location: Dict[str, Set[str]] = defaultdict(set)
        self.object_locations: Dict[str, Set[str]] = defaultdict(set)
        self.object_types: Dict[str, Set[str]] = defaultdict(set)
        self.receptacle_types: Dict[str, Set[str]] = defaultdict(set)
        self.in_receptacle: Dict[str, Set[str]] = defaultdict(set)
        self.receptacle_contents: Dict[str, Set[str]] = defaultdict(set)
        self.holds: Dict[str, Set[str]] = defaultdict(set)
        self.handempty: Set[str] = set()
        self.openable: Set[str] = set()
        self.notopenable: Set[str] = set()
        self.opened: Set[str] = set()
        self.closed: Set[str] = set()
        self.checked: Set[str] = set()
        self.cleanable: Set[str] = set()
        self.coolable: Set[str] = set()
        self.heatable: Set[str] = set()
        self.sliceable: Set[str] = set()
        self.toggleable: Set[str] = set()
        self.pickupable: Set[str] = set()
        self.isclean: Set[str] = set()
        self.iscool: Set[str] = set()
        self.ishot: Set[str] = set()
        self.ison: Set[str] = set()
        self.isoff: Set[str] = set()
        self.issliced: Set[str] = set()
        self.istoggled: Set[str] = set()
        self.cancontain: Dict[str, Set[str]] = defaultdict(set)

        for pred, args in facts:
            if pred == "atlocation" and len(args) == 2:
                self.agent_locations[args[0]] = args[1]
            elif pred == "receptacleatlocation" and len(args) == 2:
                rec, loc = args
                self.receptacle_locations[rec] = loc
                self.receptacles_by_location[loc].add(rec)
            elif pred == "objectatlocation" and len(args) == 2:
                obj, loc = args
                self.object_locations[obj].add(loc)
            elif pred == "objecttype" and len(args) == 2:
                obj, typ = args
                self.object_types[obj].add(typ)
            elif pred == "receptacletype" and len(args) == 2:
                rec, rtype = args
                self.receptacle_types[rec].add(rtype)
            elif pred == "inreceptacle" and len(args) == 2:
                obj, rec = args
                self.in_receptacle[obj].add(rec)
                self.receptacle_contents[rec].add(obj)
            elif pred == "holds" and len(args) == 2:
                agent, obj = args
                self.holds[agent].add(obj)
            elif pred == "handempty" and len(args) == 1:
                self.handempty.add(args[0])
            elif pred == "openable" and len(args) == 1:
                self.openable.add(args[0])
            elif pred == "notopenable" and len(args) == 1:
                self.notopenable.add(args[0])
            elif pred == "opened" and len(args) == 1:
                self.opened.add(args[0])
            elif pred == "closed" and len(args) == 1:
                self.closed.add(args[0])
            elif pred == "checked" and len(args) == 1:
                self.checked.add(args[0])
            elif pred == "cleanable" and len(args) == 1:
                self.cleanable.add(args[0])
            elif pred == "coolable" and len(args) == 1:
                self.coolable.add(args[0])
            elif pred == "heatable" and len(args) == 1:
                self.heatable.add(args[0])
            elif pred == "sliceable" and len(args) == 1:
                self.sliceable.add(args[0])
            elif pred == "toggleable" and len(args) == 1:
                self.toggleable.add(args[0])
            elif pred == "pickupable" and len(args) == 1:
                self.pickupable.add(args[0])
            elif pred == "isclean" and len(args) == 1:
                self.isclean.add(args[0])
            elif pred == "iscool" and len(args) == 1:
                self.iscool.add(args[0])
            elif pred == "ishot" and len(args) == 1:
                self.ishot.add(args[0])
            elif pred == "ison" and len(args) == 1:
                self.ison.add(args[0])
            elif pred == "isoff" and len(args) == 1:
                self.isoff.add(args[0])
            elif pred == "issliced" and len(args) == 1:
                self.issliced.add(args[0])
            elif pred == "istoggled" and len(args) == 1:
                self.istoggled.add(args[0])
            elif pred == "cancontain" and len(args) == 2:
                rtype, otype = args
                self.cancontain[rtype].add(otype)

        self.primary_receptacle: Dict[str, str] = {}
        for loc, recs in self.receptacles_by_location.items():
            if recs:
                self.primary_receptacle[loc] = sorted(recs)[0]


def successor(state_obj: Any) -> List[List[Dict[str, Any]]]:
    if isinstance(state_obj, dict):
        facts = _normalize_facts(state_obj.get("state", []))
    else:
        facts = _normalize_facts(state_obj)

    if not facts:
        return []

    view = IndexedState(facts)
    successors: List[Tuple[str, Set[Fact]]] = []
    seen: Set[Tuple[Fact, ...]] = set()

    def emit(action: str, new_facts: Set[Fact]) -> None:
        key = tuple(sorted(new_facts))
        if key in seen:
            return
        seen.add(key)
        successors.append((action, new_facts))

    def pair_score(obj: str, rec: str) -> Tuple[int, str, int, str]:
        rec_base, rec_idx = _split_name(rec)
        obj_base, obj_idx = _split_name(obj)
        return rec_idx, rec_base, obj_idx, obj_base

    def pair2_score(o1: str, o2: str, rec: str) -> Tuple[int, str, int, str, int, str]:
        rec_base, rec_idx = _split_name(rec)
        o1_base, o1_idx = _split_name(o1)
        o2_base, o2_idx = _split_name(o2)
        return rec_idx, rec_base, o1_idx, o1_base, o2_idx, o2_base

    # clean_object
    for agent, loc in view.agent_locations.items():
        rec = view.primary_receptacle.get(loc)
        if rec is None:
            continue
        held_objs = view.holds.get(agent, set())
        if not held_objs:
            continue
        for obj in held_objs:
            if obj not in view.cleanable:
                continue
            nf = _apply(facts, adds=[_fact("isclean", obj)])
            action = f"clean_object {agent} {loc} {rec} {obj}"
            emit(action, nf)

    # close_receptacle
    for agent, loc in view.agent_locations.items():
        for rec in view.receptacles_by_location.get(loc, set()):
            if rec not in view.openable or rec not in view.opened:
                continue
            nf = _apply(
                facts,
                adds=[_fact("closed", rec)],
                removes=[_fact("opened", rec)],
            )
            action = f"close_receptacle {agent} {loc} {rec}"
            emit(action, nf)

    # cool_object
    for agent, loc in view.agent_locations.items():
        rec = view.primary_receptacle.get(loc)
        if rec is None:
            continue
        for obj in view.holds.get(agent, set()):
            if obj not in view.coolable:
                continue
            nf = _apply(
                facts,
                adds=[_fact("iscool", obj)],
                removes=[_fact("ishot", obj)],
            )
            action = f"cool_object {agent} {loc} {rec} {obj}"
            emit(action, nf)

    # go_to_location
    for agent, start in view.agent_locations.items():
        for dest, rec in view.primary_receptacle.items():
            if dest == start:
                continue
            nf = _apply(
                facts,
                adds=[_fact("atlocation", agent, dest)],
                removes=[_fact("atlocation", agent, start)],
            )
            action = f"go_to_location {agent} {start} {dest} {rec}"
            emit(action, nf)

    # heat_object
    for agent, loc in view.agent_locations.items():
        rec = view.primary_receptacle.get(loc)
        if rec is None:
            continue
        for obj in view.holds.get(agent, set()):
            if obj not in view.heatable:
                continue
            nf = _apply(
                facts,
                adds=[_fact("ishot", obj)],
                removes=[_fact("iscool", obj)],
            )
            action = f"heat_object {agent} {loc} {rec} {obj}"
            emit(action, nf)

    # open_receptacle
    for agent, loc in view.agent_locations.items():
        for rec in view.receptacles_by_location.get(loc, set()):
            if rec not in view.openable or rec not in view.closed:
                continue
            nf = _apply(
                facts,
                adds=[_fact("opened", rec), _fact("checked", rec)],
                removes=[_fact("closed", rec)],
            )
            action = f"open_receptacle {agent} {loc} {rec}"
            emit(action, nf)

    # pickup_object_from_not_openable_receptacle
    for agent, loc in view.agent_locations.items():
        if agent not in view.handempty:
            continue
        for rec in view.receptacles_by_location.get(loc, set()):
            if rec not in view.notopenable:
                continue
            for obj in view.receptacle_contents.get(rec, set()):
                if obj not in view.pickupable:
                    continue
                nf = _apply(
                    facts,
                    adds=[_fact("holds", agent, obj)],
                    removes=[
                        _fact("inreceptacle", obj, rec),
                        _fact("handempty", agent),
                        _fact("objectatlocation", obj, loc),
                    ],
                )
                action = (
                    f"pickup_object_from_not_openable_receptacle {agent} {loc} {obj} {rec}"
                )
                emit(action, nf)

    # pickup_object_from_openable_receptacle
    for agent, loc in view.agent_locations.items():
        if agent not in view.handempty:
            continue
        for rec in view.receptacles_by_location.get(loc, set()):
            if rec not in view.openable or rec not in view.opened:
                continue
            for obj in view.receptacle_contents.get(rec, set()):
                if obj not in view.pickupable:
                    continue
                nf = _apply(
                    facts,
                    adds=[_fact("holds", agent, obj)],
                    removes=[
                        _fact("inreceptacle", obj, rec),
                        _fact("handempty", agent),
                        _fact("objectatlocation", obj, loc),
                    ],
                )
                action = (
                    f"pickup_object_from_openable_receptacle {agent} {loc} {obj} {rec}"
                )
                emit(action, nf)

    # put_object_in_openable_receptacle
    for agent, loc in view.agent_locations.items():
        for obj in view.holds.get(agent, set()):
            obj_types = view.object_types.get(obj, set())
            if not obj_types:
                continue
            for rec in view.receptacles_by_location.get(loc, set()):
                if rec not in view.openable or rec not in view.opened:
                    continue
                r_types = view.receptacle_types.get(rec, set())
                if not r_types:
                    continue
                for ot in obj_types:
                    for rt in r_types:
                        if ot not in view.cancontain.get(rt, set()):
                            continue
                        nf = _apply(
                            facts,
                            adds=[
                                _fact("inreceptacle", obj, rec),
                                _fact("objectatlocation", obj, loc),
                                _fact("handempty", agent),
                            ],
                            removes=[_fact("holds", agent, obj)],
                        )
                        action = (
                            "put_object_in_openable_receptacle "
                            f"{agent} {loc} {obj} {rec} {ot} {rt}"
                        )
                        emit(action, nf)

    # put_object_on_not_openable_receptacle
    for agent, loc in view.agent_locations.items():
        for obj in view.holds.get(agent, set()):
            obj_types = view.object_types.get(obj, set())
            if not obj_types:
                continue
            for rec in view.receptacles_by_location.get(loc, set()):
                if rec not in view.notopenable:
                    continue
                r_types = view.receptacle_types.get(rec, set())
                if not r_types:
                    continue
                for ot in obj_types:
                    for rt in r_types:
                        if ot not in view.cancontain.get(rt, set()):
                            continue
                        nf = _apply(
                            facts,
                            adds=[
                                _fact("inreceptacle", obj, rec),
                                _fact("objectatlocation", obj, loc),
                                _fact("handempty", agent),
                            ],
                            removes=[_fact("holds", agent, obj)],
                        )
                        action = (
                            "put_object_on_not_openable_receptacle "
                            f"{agent} {loc} {obj} {rec} {ot} {rt}"
                        )
                        emit(action, nf)

    # slice_object
    knife_types = {"knifetype", "butterknifetype"}
    for agent, loc in view.agent_locations.items():
        held = view.holds.get(agent, set())
        if not held:
            continue
        knives = [
            obj for obj in held if view.object_types.get(obj, set()).intersection(knife_types)
        ]
        if not knives:
            continue
        for target in view.sliceable:
            if loc not in view.object_locations.get(target, set()):
                continue
            nf = _apply(facts, adds=[_fact("issliced", target)])
            action = f"slice_object {agent} {loc} {target}"
            emit(action, nf)

    # toggle_object_off
    for agent, loc in view.agent_locations.items():
        for rec in view.receptacles_by_location.get(loc, set()):
            for obj in view.receptacle_contents.get(rec, set()):
                if obj not in view.toggleable or obj not in view.ison:
                    continue
                nf = _apply(
                    facts,
                    adds=[_fact("isoff", obj), _fact("istoggled", obj)],
                    removes=[_fact("ison", obj)],
                )
                action = f"toggle_object_off {agent} {loc} {obj} {rec}"
                emit(action, nf)

    # toggle_object_on
    for agent, loc in view.agent_locations.items():
        for rec in view.receptacles_by_location.get(loc, set()):
            for obj in view.receptacle_contents.get(rec, set()):
                if obj not in view.toggleable or obj not in view.isoff:
                    continue
                nf = _apply(
                    facts,
                    adds=[_fact("ison", obj), _fact("istoggled", obj)],
                    removes=[_fact("isoff", obj)],
                )
                action = f"toggle_object_on {agent} {loc} {obj} {rec}"
                emit(action, nf)

    # validate_clean_and_place_in_receptacle
    clean_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for obj, recs in view.in_receptacle.items():
        if obj not in view.cleanable or obj not in view.isclean:
            continue
        obj_types = view.object_types.get(obj, set())
        if not obj_types:
            continue
        for rec in recs:
            r_types = view.receptacle_types.get(rec, set())
            if not r_types:
                continue
            for ot in obj_types:
                for rt in r_types:
                    key = (ot, rt)
                    current = clean_pairs.get(key)
                    if current is None or pair_score(obj, rec) < pair_score(*current):
                        clean_pairs[key] = (obj, rec)
    for (ot, rt), (obj, rec) in clean_pairs.items():
        nf = _apply(
            facts,
            adds=[_fact("validatecleanandplace", ot, rt)],
            removes=[_fact("notvalidated")],
        )
        action = (
            "validate_clean_and_place_in_receptacle "
            f"{obj} {ot} {rec} {rt}"
        )
        emit(action, nf)

    # validate_cool_and_place_in_receptacle
    cool_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for obj, recs in view.in_receptacle.items():
        if obj not in view.coolable or obj not in view.iscool:
            continue
        obj_types = view.object_types.get(obj, set())
        if not obj_types:
            continue
        for rec in recs:
            r_types = view.receptacle_types.get(rec, set())
            if not r_types:
                continue
            for ot in obj_types:
                for rt in r_types:
                    key = (ot, rt)
                    current = cool_pairs.get(key)
                    if current is None or pair_score(obj, rec) < pair_score(*current):
                        cool_pairs[key] = (obj, rec)
    for (ot, rt), (obj, rec) in cool_pairs.items():
        nf = _apply(
            facts,
            adds=[_fact("validatecoolandplace", ot, rt)],
            removes=[_fact("notvalidated")],
        )
        action = (
            "validate_cool_and_place_in_receptacle "
            f"{obj} {ot} {rec} {rt}"
        )
        emit(action, nf)

    # validate_heat_and_place_in_receptacle
    heat_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for obj, recs in view.in_receptacle.items():
        if obj not in view.heatable or obj not in view.ishot:
            continue
        obj_types = view.object_types.get(obj, set())
        if not obj_types:
            continue
        for rec in recs:
            r_types = view.receptacle_types.get(rec, set())
            if not r_types:
                continue
            for ot in obj_types:
                for rt in r_types:
                    key = (ot, rt)
                    current = heat_pairs.get(key)
                    if current is None or pair_score(obj, rec) < pair_score(*current):
                        heat_pairs[key] = (obj, rec)
    for (ot, rt), (obj, rec) in heat_pairs.items():
        nf = _apply(
            facts,
            adds=[_fact("validateheatandplace", ot, rt)],
            removes=[_fact("notvalidated")],
        )
        action = (
            "validate_heat_and_place_in_receptacle "
            f"{obj} {ot} {rec} {rt}"
        )
        emit(action, nf)

    # validate_pick_and_place_in_receptacle
    pick_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for obj, recs in view.in_receptacle.items():
        obj_types = view.object_types.get(obj, set())
        if not obj_types:
            continue
        for rec in recs:
            r_types = view.receptacle_types.get(rec, set())
            if not r_types:
                continue
            for ot in obj_types:
                for rt in r_types:
                    key = (ot, rt)
                    current = pick_pairs.get(key)
                    if current is None or pair_score(obj, rec) < pair_score(*current):
                        pick_pairs[key] = (obj, rec)
    for (ot, rt), (obj, rec) in pick_pairs.items():
        nf = _apply(
            facts,
            adds=[_fact("validatepickandplace", ot, rt)],
            removes=[_fact("notvalidated")],
        )
        action = (
            "validate_pick_and_place_in_receptacle "
            f"{obj} {ot} {rec} {rt}"
        )
        emit(action, nf)

    # validate_pick_two_and_place_in_receptacle
    pick_two_pairs: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    for rec, objs in view.receptacle_contents.items():
        if len(objs) < 2:
            continue
        r_types = view.receptacle_types.get(rec, set())
        if not r_types:
            continue
        by_type: Dict[str, List[str]] = defaultdict(list)
        for obj in objs:
            for ot in view.object_types.get(obj, set()):
                by_type[ot].append(obj)
        for ot, obj_list in by_type.items():
            uniq = sorted(set(obj_list))
            if len(uniq) < 2:
                continue
            # choose two objects with smallest indices/names
            uniq_sorted = sorted(uniq, key=lambda o: _split_name(o)[1:] + (_split_name(o)[0],))
            o1, o2 = uniq_sorted[0], uniq_sorted[1]
            for rt in r_types:
                key = (ot, rt)
                current = pick_two_pairs.get(key)
                if current is None or pair2_score(o1, o2, rec) < pair2_score(*current):
                    pick_two_pairs[key] = (o1, o2, rec)
    for (ot, rt), (o1, o2, rec) in pick_two_pairs.items():
        nf = _apply(
            facts,
            adds=[_fact("validatepicktwoandplace", ot, rt)],
            removes=[_fact("notvalidated")],
        )
        action = (
            "validate_pick_two_and_place_in_receptacle "
            f"{o1} {o2} {ot} {rec} {rt}"
        )
        emit(action, nf)

    # validate_examine_in_light
    for agent, loc in view.agent_locations.items():
        held_objs = view.holds.get(agent, set())
        if not held_objs:
            continue
        held_types: Dict[str, Set[str]] = {obj: view.object_types.get(obj, set()) for obj in held_objs}
        for rec in view.receptacles_by_location.get(loc, set()):
            for toggle_obj in view.receptacle_contents.get(rec, set()):
                if toggle_obj not in view.toggleable or toggle_obj not in view.istoggled:
                    continue
                toggle_types = view.object_types.get(toggle_obj, set())
                if not toggle_types:
                    continue
                for held, h_types in held_types.items():
                    if not h_types:
                        continue
                    for otoggle_t in toggle_types:
                        for held_t in h_types:
                            nf = _apply(
                                facts,
                                adds=[_fact("validateexamineinlight", otoggle_t, held_t)],
                                removes=[_fact("notvalidated")],
                            )
                            action = (
                                "validate_examine_in_light "
                                f"{toggle_obj} {otoggle_t} {held} {held_t}"
                            )
                            emit(action, nf)

    return [(action, _denormalize_facts(s)) for action, s in successors]
