from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

Predicate = Dict[str, Sequence[str]]
TuplePred = Tuple[str, Tuple[str, ...]]


def _to_tuple(pred: Predicate) -> TuplePred:
    name = pred.get("predicate", "")
    args = pred.get("args", [])
    if not isinstance(args, (list, tuple)):
        args = [args]
    return name, tuple(args)


def _apply(state: Set[TuplePred], add: Iterable[TuplePred], delete: Iterable[TuplePred]) -> List[Predicate]:
    """Return a new successor state after applying add/delete effects."""
    new_state = set(state)
    for fact in delete:
        new_state.discard(fact)
    for fact in add:
        new_state.add(fact)

    return sorted(
        [{"predicate": pred, "args": list(args)} for pred, args in new_state],
        key=lambda p: (p["predicate"], tuple(p["args"]))
    )


def successor(state: List[Predicate]) -> List[List[Predicate]]:
    """
    Generate Depot-domain successor states.

    The depot domain mixes logistics-style truck movement with block-stacking
    using stationary hoists.  States are represented as a list of predicate
    dictionaries.  This function enumerates all applicable actions:

      - drive trucks between any pair of known locations
      - lift crates from pallets/crates with available hoists
      - unload crates from trucks with hoists
      - drop lifted crates onto pallets or crates
      - load lifted crates into trucks

    Each action is applied in a STRIPS-like add/delete fashion and duplicate
    successor states are removed before returning.
    """
    if not state:
        return []

    state_tuples: Set[TuplePred] = {_to_tuple(p) for p in state}

    at: Dict[str, str] = {}
    in_truck: Dict[str, str] = {}
    on_surface: Dict[str, str] = {}
    available: Set[str] = set()
    lifting: Dict[str, str] = {}
    clear: Set[str] = set()

    crates: Set[str] = set()
    hoists: Set[str] = set()
    trucks: Set[str] = set()
    pallets: Set[str] = set()
    locations: Set[str] = set()

    for pred, args in state_tuples:
        if pred == "at":
            obj, loc = args
            at[obj] = loc
            locations.add(loc)
            if obj.startswith("crate"):
                crates.add(obj)
            elif obj.startswith("hoist"):
                hoists.add(obj)
            elif obj.startswith("truck"):
                trucks.add(obj)
            elif obj.startswith("pallet"):
                pallets.add(obj)

        elif pred == "in":
            crate, truck = args
            in_truck[crate] = truck
            crates.add(crate)
            trucks.add(truck)

        elif pred == "on":
            crate, surface = args
            on_surface[crate] = surface
            crates.add(crate)
            if surface.startswith("crate"):
                crates.add(surface)
            elif surface.startswith("pallet"):
                pallets.add(surface)

        elif pred == "available":
            hoist = args[0]
            available.add(hoist)
            hoists.add(hoist)

        elif pred == "lifting":
            hoist, crate = args
            lifting[hoist] = crate
            hoists.add(hoist)
            crates.add(crate)

        elif pred == "clear":
            clear.add(args[0])

    successors: List[Tuple[str, List[Predicate]]] = []
    successor_keys: Set[frozenset] = set()

    def append_successor(action: str, add: Iterable[TuplePred], delete: Iterable[TuplePred]) -> None:
        succ = _apply(state_tuples, add, delete)
        key = frozenset(_to_tuple(p) for p in succ)
        if key not in successor_keys:
            successor_keys.add(key)
            successors.append((action, succ))

    # --- Drive trucks between locations ---
    for truck in sorted(trucks):
        current_loc = at.get(truck)
        if not current_loc:
            continue
        for dest in sorted(locations):
            action = f"drive {truck} {current_loc} {dest}"
            add = [("at", (truck, dest))]
            delete = [("at", (truck, current_loc))]
            append_successor(action, add, delete)

    # --- Lift crates from stacks/pallets ---
    for crate, surface in on_surface.items():
        loc = at.get(crate)
        if not loc or crate not in clear:
            continue

        # Ensure supporting surface is anchored at the same location.
        surface_loc = at.get(surface)
        if not surface_loc or surface_loc != loc:
            continue

        for hoist in hoists:
            if hoist not in available:
                continue
            if at.get(hoist) != loc:
                continue
            action = f"lift {hoist} {crate} {surface} {loc}"
            add = [
                ("lifting", (hoist, crate)),
                ("clear", (surface,))
            ]
            delete = [
                ("available", (hoist,)),
                ("on", (crate, surface)),
                ("clear", (crate,)),
                ("at", (crate, loc)),
            ]
            append_successor(action, add, delete)

    # --- Unload crates from trucks (hoist picks crate) ---
    for crate, truck in in_truck.items():
        truck_loc = at.get(truck)
        if not truck_loc:
            continue
        for hoist in hoists:
            if hoist not in available:
                continue
            if at.get(hoist) != truck_loc:
                continue
            action = f"unload {hoist} {crate} {truck} {truck_loc}"
            add = [("lifting", (hoist, crate))]
            delete = [
                ("available", (hoist,)),
                ("in", (crate, truck)),
            ]
            append_successor(action, add, delete)

    # --- Drop lifted crates onto pallets or crates ---
    for hoist, crate in lifting.items():
        hoist_loc = at.get(hoist)
        if not hoist_loc:
            continue

        for surface in list(clear):
            if surface == crate:
                continue
            surface_loc = at.get(surface)
            if not surface_loc or surface_loc != hoist_loc:
                continue
            action = f"drop {hoist} {crate} {surface} {hoist_loc}"
            add = [
                ("on", (crate, surface)),
                ("clear", (crate,)),
                ("available", (hoist,)),
                ("at", (crate, hoist_loc)),
            ]
            delete = [
                ("lifting", (hoist, crate)),
                ("clear", (surface,)),
            ]
            append_successor(action, add, delete)

    # --- Load lifted crates into trucks ---
    for hoist, crate in lifting.items():
        hoist_loc = at.get(hoist)
        if not hoist_loc:
            continue
        for truck in trucks:
            if at.get(truck) != hoist_loc:
                continue
            action = f"load {hoist} {crate} {truck} {hoist_loc}"
            add = [
                ("in", (crate, truck)),
                ("available", (hoist,)),
            ]
            delete = [
                ("lifting", (hoist, crate)),
                ("at", (crate, hoist_loc)),
                ("clear", (crate,)),
            ]
            append_successor(action, add, delete)

    return successors
