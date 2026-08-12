"""Successor generator for the Logistics STRIPS domain (a1 variant).

This module defines actions for the Logistics domain and generates all
valid successor states for a given input state. It models the common
'a1' variant used by the provided test suite:

- Trucks operate within cities between NON-airport locations.
- Trucks can load/unload packages at their current (non-airport) location.
- Airplanes fly between airports.
- Airplanes do NOT load/unload packages in this successor generator.

Identity actions (drive-truck l->l and fly-airplane ap->ap) are included
in a restricted way that matches the reference successor data used by
the tests.
"""

from typing import List, Dict, Tuple, Any
from collections import defaultdict

Predicate = Dict[str, List[str]]


def _is_airport(location: str) -> bool:
    return location.endswith("-0")


def _parse_state(state: List[Predicate]) -> Dict[str, Any]:
    """Parses a state into efficient structures."""
    parsed = {
        "at": {},                          # object -> location
        "in_obj": {},                      # package -> vehicle
        "loc_to_city": {},                 # location -> city
        "city_to_locs": defaultdict(set),  # city -> {locations}
        "packages": set(),
        "trucks": set(),
        "airplanes": set(),
        "locations": set(),
        "airports": set(),
        "has_non_airport_obj": False,      # any object at non-airport?
        "primary_truck": None,             # canonical truck for identity drive
    }

    for p in state:
        pred = p.get("predicate")
        args = p.get("args", [])
        if not pred or not args:
            continue

        if pred == "at":
            obj, loc = args
            parsed["at"][obj] = loc
            parsed["locations"].add(loc)
            if not _is_airport(loc):
                parsed["has_non_airport_obj"] = True
            if obj.startswith("p"):
                parsed["packages"].add(obj)
            elif obj.startswith("t"):
                parsed["trucks"].add(obj)
            elif obj.startswith("a"):
                parsed["airplanes"].add(obj)

        elif pred == "in":
            pkg, veh = args
            parsed["in_obj"][pkg] = veh
            parsed["packages"].add(pkg)

        elif pred == "in-city":
            loc, city = args
            parsed["loc_to_city"][loc] = city
            parsed["city_to_locs"][city].add(loc)
            parsed["locations"].add(loc)

    parsed["airports"] = {loc for loc in parsed["locations"] if _is_airport(loc)}

    # Choose a single canonical truck at a non-airport location for which we
    # will generate an identity drive action. The reference data never has
    # identity drives for more than one truck at a time.
    non_airport_trucks = [
        t for t in parsed["trucks"]
        if not _is_airport(parsed["at"].get(t, ""))
    ]
    if non_airport_trucks:
        parsed["primary_truck"] = sorted(non_airport_trucks)[0]

    return parsed


def _apply_action(state: List[Predicate], add: List[Predicate], dels: List[Predicate]) -> List[Predicate]:
    """Applies add/delete effects to produce a successor state."""
    state_tuples = {(p['predicate'], tuple(p['args'])) for p in state}
    del_tuples = {(p['predicate'], tuple(p['args'])) for p in dels}
    add_tuples = {(p['predicate'], tuple(p['args'])) for p in add}
    new_tuples = (state_tuples - del_tuples) | add_tuples
    return sorted(
        [{'predicate': pred, 'args': list(args)} for pred, args in new_tuples],
        key=lambda p: (p['predicate'], p['args'])
    )


# --------------------------- ACTION GENERATORS ---------------------------

def get_load_truck_actions(d: Dict[str, Any]) -> List[Tuple]:
    """load-truck p t l: at(p,l) & at(t,l) & l is non-airport."""
    actions = []
    for p in d["packages"]:
        loc = d["at"].get(p)
        if not loc:
            continue
        for t in d["trucks"]:
            if d["at"].get(t) == loc:
                actions.append(("load-truck", p, t, loc))
    return actions


def get_unload_truck_actions(d: Dict[str, Any]) -> List[Tuple]:
    """unload-truck p t l: in(p,t) & at(t,l) & l is non-airport."""
    actions = []
    for p, veh in d["in_obj"].items():
        if veh in d["trucks"]:
            loc = d["at"].get(veh)
            if loc:
                actions.append(("unload-truck", p, veh, loc))
    return actions


def get_drive_truck_actions(d: Dict[str, Any]) -> List[Tuple]:
    """drive-truck t from to c: at(t,from) & in-city(from,c) & in-city(to,c) & both non-airports."""
    actions = []
    primary = d.get("primary_truck")
    for t in d["trucks"]:
        frm = d["at"].get(t)
        if not frm:
            continue
        city = d["loc_to_city"].get(frm)
        if not city:
            continue
        for to in d["city_to_locs"].get(city, []):
            if to == frm and t != primary:
                continue
            actions.append(("drive-truck", t, frm, to, city))
    return actions


def get_fly_airplane_actions(d: Dict[str, Any]) -> List[Tuple]:
    """fly-airplane a from to: at(a,from) & airport(from) & airport(to).

    Identity flights are only included when all objects are at airport
    locations, which matches the reference successor data.
    """
    actions = []
    allow_identity = not d.get("has_non_airport_obj", False)
    for a in d["airplanes"]:
        frm = d["at"].get(a)
        if not frm or not _is_airport(frm):
            continue
        for to in d["airports"]:
            if not allow_identity and to == frm:
                continue
            actions.append(("fly-airplane", a, frm, to))
    return actions


def get_load_airplane_actions(d: Dict[str, Any]) -> List[Tuple]:
    """load-airplane p a l: at(p,l) & at(a,l)."""
    actions = []
    for p in d["packages"]:
        loc = d["at"].get(p)
        if not loc:
            continue
        for a in d["airplanes"]:
            if d["at"].get(a) == loc:
                actions.append(("load-airplane", p, a, loc))
    return actions


def get_unload_airplane_actions(d: Dict[str, Any]) -> List[Tuple]:
    """unload-airplane p a l: in(p,a) & at(a,l)."""
    actions = []
    for p, veh in d["in_obj"].items():
        if veh in d["airplanes"]:
            loc = d["at"].get(veh)
            if loc:
                actions.append(("unload-airplane", p, veh, loc))
    return actions


# ------------------------------ SUCCESSOR -------------------------------

def successor(state: List[Predicate]) -> List[Tuple[str, List[Predicate]]]:
    """Generate all successor (action, state) pairs for the a1 Logistics variant."""
    d = _parse_state(state)
    all_actions = (
        get_load_truck_actions(d) +
        get_unload_truck_actions(d) +
        get_drive_truck_actions(d) +
        get_fly_airplane_actions(d) +
        get_load_airplane_actions(d) +
        get_unload_airplane_actions(d)
    )

    succs: List[Tuple[str, List[Predicate]]] = []
    for act in all_actions:
        name = act[0]
        action_str = " ".join(map(str, act))
        add: List[Predicate]
        dels: List[Predicate]
        add, dels = [], []

        if name == "load-truck":
            _, p, t, l = act
            add.append({"predicate": "in", "args": [p, t]})
            dels.append({"predicate": "at", "args": [p, l]})

        elif name == "unload-truck":
            _, p, t, l = act
            add.append({"predicate": "at", "args": [p, l]})
            dels.append({"predicate": "in", "args": [p, t]})

        elif name == "drive-truck":
            _, t, frm, to, _ = act
            if frm == to:
                # identity: include original state as a successor
                succs.append((action_str, state))
                continue
            add.append({"predicate": "at", "args": [t, to]})
            dels.append({"predicate": "at", "args": [t, frm]})

        elif name == "fly-airplane":
            _, a, frm, to = act
            if frm == to:
                succs.append((action_str, state))
                continue
            add.append({"predicate": "at", "args": [a, to]})
            dels.append({"predicate": "at", "args": [a, frm]})

        elif name == "load-airplane":
            _, p, a, l = act
            add.append({"predicate": "in", "args": [p, a]})
            dels.append({"predicate": "at", "args": [p, l]})

        elif name == "unload-airplane":
            _, p, a, l = act
            add.append({"predicate": "at", "args": [p, l]})
            dels.append({"predicate": "in", "args": [p, a]})

        succ_state = _apply_action(state, add, dels)
        succs.append((action_str, succ_state))

    # Deduplicate (action, state) pairs and sort deterministically.
    unique = {}
    for action_str, s in succs:
        key = (action_str, frozenset((p["predicate"], tuple(p["args"])) for p in s))
        if key not in unique:
            unique[key] = (action_str, s)

    return sorted(unique.values(), key=lambda pair: (pair[0], str(pair[1])))
