from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

Predicate = Tuple[str, Tuple[str, ...]]
State = List[Dict[str, Sequence[str]]]


def _entry_to_tuple(entry: Dict[str, Sequence[str]]) -> Predicate:
    """Convert a predicate dict into a tuple we can place in a set."""
    return entry.get("predicate"), tuple(entry.get("args", []) or [])


def _tuples_to_state(predicates: Set[Predicate]) -> State:
    """Convert the internal tuple representation back to the test format."""
    return [
        {"predicate": pred, "args": list(args)}
        for pred, args in sorted(predicates, key=lambda item: (item[0], item[1]))
    ]


def successor(state: State) -> List[Tuple[str, State]]:
    """
    Compute all applicable successor states for the gold miner domain.

    Each returned state is represented in the same format as the input `state`:
    a list of {predicate, args} dictionaries.
    """
    state_set: Set[Predicate] = {_entry_to_tuple(p) for p in state}
    robot_loc = next((args[0] for pred, args in state_set if pred == "robot-at" and args), None)
    if robot_loc is None:
        return []

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for pred, args in state_set:
        if pred == "connected" and len(args) == 2:
            adjacency[args[0]].add(args[1])

    arm_empty = ("arm-empty", ()) in state_set
    holds_laser = ("holds-laser", ()) in state_set
    holds_bomb = ("holds-bomb", ()) in state_set

    # Map each distinct successor state to a single, canonical action.
    # If multiple actions lead to the same state, keep the lexicographically
    # smallest action string to match the expected test behavior.
    result_map: Dict[Tuple[Predicate, ...], Tuple[str, State]] = {}

    def emit(
        action: str,
        add: Optional[Set[Predicate]] = None,
        remove: Optional[Set[Predicate]] = None,
    ) -> None:
        """Apply add/remove sets to the base state and record (action, state)."""
        new_state = set(state_set)
        if remove:
            new_state.difference_update(remove)
        if add:
            new_state.update(add)
        key = tuple(sorted(new_state))
        existing = result_map.get(key)
        if existing is None or action < existing[0]:
            result_map[key] = (action, _tuples_to_state(new_state))

    # Move into any adjacent clear cell.
    for dest in sorted(adjacency.get(robot_loc, [])):
        if ("clear", (dest,)) in state_set:
            emit(
                action=f"move {robot_loc} {dest}",
                add={("robot-at", (dest,))},
                remove={("robot-at", (robot_loc,))},
            )

    # Pick up bombs or lasers when the robot's arm is free.
    if arm_empty and ("bomb-at", (robot_loc,)) in state_set:
        emit(
            action=f"pickup-bomb {robot_loc}",
            add={("holds-bomb", ())},
            remove={("arm-empty", ())},
        )

    if arm_empty and ("laser-at", (robot_loc,)) in state_set:
        emit(
            action=f"pickup-laser {robot_loc}",
            add={("holds-laser", ())},
            remove={("arm-empty", ()), ("laser-at", (robot_loc,))},
        )

    # Put down a laser at the current location.
    if holds_laser:
        emit(
            action=f"putdown-laser {robot_loc}",
            add={("laser-at", (robot_loc,)), ("arm-empty", ())},
            remove={("holds-laser", ())},
        )

    # Fire the laser at any adjacent cell. Destroy rocks if present.
    if holds_laser:
        for target in sorted(adjacency.get(robot_loc, [])):
            to_remove: Set[Predicate] = set()
            if ("soft-rock-at", (target,)) in state_set:
                to_remove.add(("soft-rock-at", (target,)))
            if ("hard-rock-at", (target,)) in state_set:
                to_remove.add(("hard-rock-at", (target,)))
            to_add: Set[Predicate] = set()
            if to_remove:
                to_add.add(("clear", (target,)))
            emit(
                action=f"fire-laser {robot_loc} {target}",
                add=to_add,
                remove=to_remove,
            )

    # Detonate bombs to clear adjacent soft rock cells.
    if holds_bomb:
        for target in sorted(adjacency.get(robot_loc, [])):
            if ("soft-rock-at", (target,)) in state_set:
                emit(
                    action=f"detonate-bomb {robot_loc} {target}",
                    add={("clear", (target,)), ("arm-empty", ())},
                    remove={("soft-rock-at", (target,)), ("holds-bomb", ())},
                )

    return list(result_map.values())
