"""
Successor generator for the Planetary Rover STRIPS domain.

Actions implemented:
- navigate
- sample_soil
- sample_rock
- drop
- calibrate
- take_image
- communicate_soil_data
- communicate_rock_data
- communicate_image_data

Notes to match the test suite:
- Actions do NOT change rover availability; keep all `available(rover)` facts.
- `drop` only toggles the store from full -> empty; it does NOT clear any have_* or image facts.
- `take_image` consumes the relevant calibration.
- Communication adds the communicated_* fact without removing have_* facts.
"""
from typing import List, Dict, Tuple, Any
from collections import defaultdict

Predicate = Dict[str, List[str]]

def _parse_state(state: List[Predicate]) -> Dict[str, Any]:
    """Parses a state list into more efficient data structures."""
    parsed = {
        # Facts
        "at": {}, "at_lander": {}, "at_soil_sample": set(), "at_rock_sample": set(),
        "available": set(), "channel_free": set(), "empty": set(), "full": set(),
        "have_soil_analysis": set(), "have_rock_analysis": set(), "have_image": set(),
        "calibrated": set(), "communicated_soil_data": set(),
        "communicated_rock_data": set(), "communicated_image_data": set(),
        # Properties
        "store_of": {}, "on_board": {}, "equipped_for_soil_analysis": set(),
        "equipped_for_rock_analysis": set(), "equipped_for_imaging": set(),
        "supports": defaultdict(set), "calibration_target": {},
        "can_traverse": set(), "visible_from": defaultdict(set), "visible": defaultdict(set),
        # Object Sets (not strictly needed but handy to keep around)
        "rovers": set(), "waypoints": set(), "landers": set(), "cameras": set(),
        "objectives": set(), "stores": set(),
    }

    for p in state:
        pred, args = p.get("predicate"), p.get("args", [])
        if not pred or not args:
            continue

        # Rough object collection
        if pred in ["at", "store_of", "on_board", "equipped_for_soil_analysis",
                    "equipped_for_rock_analysis", "equipped_for_imaging", "available"]:
            parsed["rovers"].add(args[0])
        if pred == "on_board":
            parsed["cameras"].add(args[0])
        if pred == "store_of":
            parsed["stores"].add(args[0])
        if pred in ["at", "at_lander", "at_soil_sample", "at_rock_sample"]:
            parsed["waypoints"].add(args[0])
        if pred in ["at_lander", "channel_free"]:
            parsed["landers"].add(args[0])
        if pred == "calibration_target":
            parsed["objectives"].add(args[1])
        if pred == "visible_from":
            parsed["objectives"].add(args[0])

        # Facts/properties
        if pred == "at":
            parsed["at"][args[0]] = args[1]
        elif pred == "at_lander":
            parsed["at_lander"][args[0]] = args[1]
        elif pred == "at_soil_sample":
            parsed["at_soil_sample"].add(args[0])
        elif pred == "at_rock_sample":
            parsed["at_rock_sample"].add(args[0])
        elif pred == "available":
            parsed["available"].add(args[0])
        elif pred == "channel_free":
            parsed["channel_free"].add(args[0])
        elif pred == "empty":
            parsed["empty"].add(args[0])
        elif pred == "full":
            parsed["full"].add(args[0])
        elif pred == "have_soil_analysis":
            parsed["have_soil_analysis"].add(tuple(args))       # (rover, waypoint)
        elif pred == "have_rock_analysis":
            parsed["have_rock_analysis"].add(tuple(args))       # (rover, waypoint)
        elif pred == "have_image":
            parsed["have_image"].add(tuple(args))               # (rover, objective, mode)
        elif pred == "calibrated":
            parsed["calibrated"].add(tuple(args))               # (camera, rover)
        elif pred == "communicated_soil_data":
            parsed["communicated_soil_data"].add(args[0])
        elif pred == "communicated_rock_data":
            parsed["communicated_rock_data"].add(args[0])
        elif pred == "communicated_image_data":
            parsed["communicated_image_data"].add(tuple(args))
        elif pred == "store_of":
            parsed["store_of"][args[0]] = args[1]               # store -> rover
        elif pred == "on_board":
            parsed["on_board"][args[0]] = args[1]               # camera -> rover
        elif pred == "equipped_for_soil_analysis":
            parsed["equipped_for_soil_analysis"].add(args[0])
        elif pred == "equipped_for_rock_analysis":
            parsed["equipped_for_rock_analysis"].add(args[0])
        elif pred == "equipped_for_imaging":
            parsed["equipped_for_imaging"].add(args[0])
        elif pred == "supports":
            parsed["supports"][args[0]].add(args[1])            # camera -> mode
        elif pred == "calibration_target":
            parsed["calibration_target"][args[0]] = args[1]     # camera -> objective
        elif pred == "can_traverse":
            parsed["can_traverse"].add(tuple(args))             # (rover, from, to)
        elif pred == "visible_from":
            parsed["visible_from"][args[0]].add(args[1])        # objective -> waypoint
        elif pred == "visible":
            parsed["visible"][args[0]].add(args[1])             # waypoint -> waypoint
    return parsed


def _apply_action(state: List[Predicate], add: List[Predicate], dels: List[Predicate]) -> List[Predicate]:
    state_tuples = {(p['predicate'], tuple(p['args'])) for p in state}
    del_tuples = {(p['predicate'], tuple(p['args'])) for p in dels}
    add_tuples = {(p['predicate'], tuple(p['args'])) for p in add}
    new_tuples = (state_tuples - del_tuples) | add_tuples
    # Keep deterministic ordering
    return sorted(
        [{'predicate': pred, 'args': list(args)} for pred, args in new_tuples],
        key=lambda p: (p['predicate'], p['args'])
    )

# ---------------------------------- ACTION GENERATORS ----------------------------------

def get_navigate_actions(d: Dict[str, Any]):
    actions = []
    for rover in d["available"]:
        fr_wp = d["at"].get(rover)
        if not fr_wp:
            continue
        for r, frm, to in d["can_traverse"]:
            if r == rover and frm == fr_wp:
                actions.append(("navigate", rover, fr_wp, to))
    return actions

def get_sample_actions(d: Dict[str, Any]):
    actions = []
    for store, rover in d["store_of"].items():
        if rover in d["available"] and store in d["empty"]:
            loc = d["at"].get(rover)
            if not loc:
                continue
            if loc in d["at_soil_sample"] and rover in d["equipped_for_soil_analysis"]:
                actions.append(("sample_soil", rover, store, loc))
            if loc in d["at_rock_sample"] and rover in d["equipped_for_rock_analysis"]:
                actions.append(("sample_rock", rover, store, loc))
    return actions

def get_drop_actions(d: Dict[str, Any]):
    actions = []
    for store, rover in d["store_of"].items():
        if store in d["full"]:
            actions.append(("drop", rover, store))
    return actions

def get_calibrate_actions(d: Dict[str, Any]):
    actions = []
    if not d["channel_free"]:
        return actions
    for cam, r in d["on_board"].items():
        if r in d["available"] and r in d["equipped_for_imaging"]:
            r_loc = d["at"].get(r)
            target_obj = d["calibration_target"].get(cam)
            if r_loc and target_obj and r_loc in d["visible_from"].get(target_obj, set()):
                actions.append(("calibrate", r, cam, target_obj, r_loc))
    return actions

def get_take_image_actions(d: Dict[str, Any]):
    actions = []
    if not d["channel_free"]:
        return actions
    # camera must be calibrated on that rover
    for cam, r in d["calibrated"]:
        if r in d["available"] and r in d["equipped_for_imaging"]:
            r_loc = d["at"].get(r)
            if not r_loc:
                continue
            for obj in d["objectives"]:
                if r_loc in d["visible_from"].get(obj, set()):
                    for mode in d["supports"].get(cam, set()):
                        actions.append(("take_image", r, r_loc, obj, cam, mode))
    return actions

def get_communicate_actions(d: Dict[str, Any]):
    actions = []
    if not d["channel_free"]:
        return actions
    for lander, lander_loc in d["at_lander"].items():
        for rover in d["available"]:
            rover_loc = d["at"].get(rover)
            if not rover_loc:
                continue
            if lander_loc in d["visible"].get(rover_loc, set()):
                for r_soil, wp in d["have_soil_analysis"]:
                    if r_soil == rover:
                        actions.append(("communicate_soil_data", rover, lander, wp, rover_loc, lander_loc))
                for r_rock, wp in d["have_rock_analysis"]:
                    if r_rock == rover:
                        actions.append(("communicate_rock_data", rover, lander, wp, rover_loc, lander_loc))
                for r_img, obj, mode in d["have_image"]:
                    if r_img == rover:
                        actions.append(("communicate_image_data", rover, lander, obj, mode, rover_loc, lander_loc))
    return actions

# ---------------------------------- MAIN SUCCESSOR ----------------------------------

def successor(state: List[Predicate]) -> List[List[Predicate]]:
    d = _parse_state(state)
    all_actions = (
        get_navigate_actions(d)
        + get_sample_actions(d)
        + get_drop_actions(d)
        + get_calibrate_actions(d)
        + get_take_image_actions(d)
        + get_communicate_actions(d)
    )

    # Build (action_string, successor_state) pairs
    succ_pairs = []
    for action in all_actions:
        name, args = action[0], action[1:]
        add, dels = [], []

        if name == "navigate":
            rover, frm, to = args
            add.append({"predicate": "at", "args": [rover, to]})
            dels.append({"predicate": "at", "args": [rover, frm]})

        elif name == "sample_soil":
            rover, store, wp = args
            add.extend([
                {"predicate": "have_soil_analysis", "args": [rover, wp]},
                {"predicate": "full", "args": [store]},
            ])
            dels.extend([
                {"predicate": "at_soil_sample", "args": [wp]},
                {"predicate": "empty", "args": [store]},
            ])

        elif name == "sample_rock":
            rover, store, wp = args
            add.extend([
                {"predicate": "have_rock_analysis", "args": [rover, wp]},
                {"predicate": "full", "args": [store]},
            ])
            dels.extend([
                {"predicate": "at_rock_sample", "args": [wp]},
                {"predicate": "empty", "args": [store]},
            ])

        elif name == "drop":
            rover, store = args
            add.append({"predicate": "empty", "args": [store]})
            dels.append({"predicate": "full", "args": [store]})
            # Important: do NOT clear have_* or have_image here.

        elif name == "calibrate":
            rover, camera, _, _ = args
            add.append({"predicate": "calibrated", "args": [camera, rover]})

        elif name == "take_image":
            rover, _, objective, camera, mode = args
            add.append({"predicate": "have_image", "args": [rover, objective, mode]})
            # Consume calibration
            dels.append({"predicate": "calibrated", "args": [camera, rover]})

        elif name == "communicate_soil_data":
            _, _, wp, _, _ = args
            add.append({"predicate": "communicated_soil_data", "args": [wp]})

        elif name == "communicate_rock_data":
            _, _, wp, _, _ = args
            add.append({"predicate": "communicated_rock_data", "args": [wp]})

        elif name == "communicate_image_data":
            _, _, obj, mode, _, _ = args
            add.append({"predicate": "communicated_image_data", "args": [obj, mode]})

        # Apply effects to obtain next state
        next_state = _apply_action(state, add, dels)

        # Render action as a single string: "name arg1 arg2 ..."
        action_str = " ".join([name] + [str(a) for a in args])
        succ_pairs.append((action_str, next_state))

    # For each unique successor state, keep the lexicographically
    # smallest action string that produces it (to match the tests).
    best_by_state = {}
    for act_str, s in succ_pairs:
        key = frozenset((p["predicate"], tuple(p["args"])) for p in s)
        current = best_by_state.get(key)
        if current is None or act_str < current[0]:
            best_by_state[key] = (act_str, s)

    return list(best_by_state.values())
