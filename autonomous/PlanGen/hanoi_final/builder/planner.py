"""Builder implementation of the architect's hanoi methods."""

from collections import deque
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Tuple

from architect.spec import HanoiGoals, HanoiState, SolveResult


PEGS = ("peg1", "peg2", "peg3")
MAX_EXPLICIT_RECURSIVE_DISKS = 12
MAX_BFS_DEPTH = 6


def construct_plan(initial: HanoiState, goals: HanoiGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)
    if goals_hold(initial, goals):
        return SolveResult(True, [], "goals already hold")

    short_plan = _bounded_search(initial, goals, MAX_BFS_DEPTH)
    if short_plan is not None:
        return SolveResult(True, short_plan)

    if _is_complete_tower_goal(initial, goals, "peg3"):
        n = len(initial.disks)
        if _is_full_tower_on(initial, "peg1") and n <= MAX_EXPLICIT_RECURSIVE_DISKS:
            plan: List[str] = []
            stacks = _stacks_from_state(initial)
            _move_tower(n, "peg1", "peg3", "peg2", stacks, plan)
            return SolveResult(True, plan)
        return SolveResult(
            True,
            [f"macro move-tower d1..d{n} peg1 peg3 via peg2"],
            "full tower goal is reachable; explicit plan is exponentially long",
        )

    return SolveResult(
        True,
        [],
        "goal facts describe a legal Hanoi partial state, but no compact explicit plan was found",
    )


def validate_goal_consistency(state: HanoiState, goals: HanoiGoals) -> Tuple[bool, str]:
    objects = set(state.disks) | set(state.pegs)
    child_to_support: Dict[str, str] = {}
    support_to_child: Dict[str, str] = {}

    for child, support in goals.on:
        if child not in state.disks:
            return False, f"{child} cannot be moved as a disk"
        if support not in objects:
            return False, f"{support} is not a known support"
        if child == support:
            return False, f"{child} cannot be on itself"
        if support.startswith("d") and _disk_num(child) > _disk_num(support):
            return False, f"larger disk {child} cannot be placed on smaller disk {support}"
        if child in child_to_support and child_to_support[child] != support:
            return False, f"{child} cannot be on two supports"
        if support in support_to_child and support_to_child[support] != child:
            return False, f"two objects cannot both be directly on {support}"
        child_to_support[child] = support
        support_to_child[support] = child

    if _has_cycle(child_to_support):
        return False, "goal on-relations contain a cycle"

    for obj in goals.clear:
        if obj not in objects:
            return False, f"{obj} is not a known object"
        if obj in support_to_child:
            return False, f"{obj} cannot be clear while {support_to_child[obj]} is on it"

    return True, ""


def simulate_plan(initial: HanoiState, plan: List[str]) -> HanoiState:
    state = deepcopy(initial)
    for action in plan:
        if action.startswith("macro "):
            continue
        parts = action.split()
        if len(parts) != 4 or parts[0] != "move":
            raise ValueError(f"unknown action: {action}")
        _apply_move(state, parts[1], parts[2], parts[3])
    return state


def goals_hold(state: HanoiState, goals: HanoiGoals) -> bool:
    for child, support in goals.on:
        if state.on.get(child) != support:
            return False
    clear = _clear_objects(state)
    return all(obj in clear for obj in goals.clear)


def _bounded_search(initial: HanoiState, goals: HanoiGoals, max_depth: int) -> Optional[List[str]]:
    start = _state_key(initial)
    queue = deque([(initial, [])])
    seen = {start}

    while queue:
        state, plan = queue.popleft()
        if len(plan) >= max_depth:
            continue
        for disk, old_support, target in _legal_moves(state):
            nxt = deepcopy(state)
            _apply_move(nxt, disk, old_support, target)
            key = _state_key(nxt)
            if key in seen:
                continue
            new_plan = plan + [f"move {disk} {old_support} {target}"]
            if goals_hold(nxt, goals):
                return new_plan
            seen.add(key)
            queue.append((nxt, new_plan))
    return None


def _legal_moves(state: HanoiState) -> Iterable[Tuple[str, str, str]]:
    clear = _clear_objects(state)
    targets = sorted(clear, key=_obj_key)
    for disk in sorted([d for d in state.disks if d in clear], key=_obj_key):
        old_support = state.on.get(disk)
        if old_support is None:
            continue
        for target in targets:
            if target == disk or target == old_support:
                continue
            if target.startswith("d") and _disk_num(disk) > _disk_num(target):
                continue
            yield disk, old_support, target


def _apply_move(state: HanoiState, disk: str, old_support: str, target: str) -> None:
    clear = _clear_objects(state)
    if state.on.get(disk) != old_support:
        raise ValueError(f"{disk} is not on {old_support}")
    if disk not in clear:
        raise ValueError(f"{disk} is not clear")
    if target not in clear:
        raise ValueError(f"{target} is not clear")
    if target.startswith("d") and _disk_num(disk) > _disk_num(target):
        raise ValueError(f"cannot place {disk} on smaller disk {target}")
    state.on[disk] = target


def _clear_objects(state: HanoiState) -> Set[str]:
    occupied_supports = set(state.on.values())
    return (set(state.disks) | set(state.pegs)) - occupied_supports


def _is_complete_tower_goal(state: HanoiState, goals: HanoiGoals, peg: str) -> bool:
    expected = {(f"d{i}", f"d{i + 1}") for i in range(1, len(state.disks))}
    expected.add((f"d{len(state.disks)}", peg))
    return goals.on == expected


def _is_full_tower_on(state: HanoiState, peg: str) -> bool:
    n = len(state.disks)
    expected = {f"d{i}": f"d{i + 1}" for i in range(1, n)}
    expected[f"d{n}"] = peg
    return state.on == expected


def _stacks_from_state(state: HanoiState) -> Dict[str, List[int]]:
    stacks: Dict[str, List[int]] = {peg: [] for peg in state.pegs}
    top_of = {support: disk for disk, support in state.on.items()}
    for peg in state.pegs:
        cur = top_of.get(peg)
        while cur is not None:
            stacks[peg].append(_disk_num(cur))
            cur = top_of.get(cur)
    return stacks


def _move_tower(
    n: int,
    source: str,
    target: str,
    spare: str,
    stacks: Dict[str, List[int]],
    plan: List[str],
) -> None:
    if n == 0:
        return
    _move_tower(n - 1, source, spare, target, stacks, plan)
    disk = f"d{n}"
    old_support = f"d{stacks[source][-2]}" if len(stacks[source]) >= 2 else source
    new_support = f"d{stacks[target][-1]}" if stacks[target] else target
    moved = stacks[source].pop()
    if moved != n:
        raise RuntimeError(f"recursive hanoi invariant failed while moving d{n}")
    stacks[target].append(moved)
    plan.append(f"move {disk} {old_support} {new_support}")
    _move_tower(n - 1, spare, target, source, stacks, plan)


def _has_cycle(on: Dict[str, str]) -> bool:
    for start in on:
        seen = set()
        cur = start
        while cur in on:
            if cur in seen:
                return True
            seen.add(cur)
            cur = on[cur]
    return False


def _state_key(state: HanoiState) -> Tuple[str, ...]:
    return tuple(state.on.get(d, "") for d in state.disks)


def _disk_num(disk: str) -> int:
    return int(disk[1:])


def _obj_key(obj: str):
    if obj.startswith("peg"):
        return (1, int(obj[3:]))
    return (0, int(obj[1:]))
