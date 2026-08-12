"""Builder implementation of the architect's grid methods."""

from heapq import heappop, heappush
from itertools import count
from typing import Dict, Iterable, List, Optional, Set, Tuple

from architect.spec import GridGoals, GridState, SolveResult


SEARCH_LIMIT = 400000


def construct_plan(initial: GridState, goals: GridGoals) -> SolveResult:
    impossible = _static_impossibility(initial, goals)
    if impossible:
        return SolveResult(False, [], impossible)
    if goals_hold(initial, goals):
        return SolveResult(True, [])

    if sum(1 for pred, _, _ in goals.atoms if pred == "key-at") >= 2:
        plan = _construct_key_delivery_plan(initial, goals)
        if plan is not None:
            return SolveResult(True, plan)

    queue: List[Tuple[int, int, int, GridState, List[str]]] = []
    serial = count()
    heappush(queue, (_heuristic(initial, goals), 0, next(serial), initial, []))
    best_depth: Dict[GridState, int] = {initial: 0}
    expansions = 0

    while queue and expansions < SEARCH_LIMIT:
        _, depth, _, state, plan = heappop(queue)
        if depth != best_depth.get(state):
            continue
        expansions += 1

        for action, nxt in _successors(state, goals):
            new_depth = depth + 1
            if new_depth >= best_depth.get(nxt, 10**9):
                continue
            new_plan = plan + [action]
            if goals_hold(nxt, goals):
                return SolveResult(True, new_plan)
            best_depth[nxt] = new_depth
            heappush(queue, (new_depth + _heuristic(nxt, goals), new_depth, next(serial), nxt, new_plan))

    return SolveResult(False, [], "search exhausted without reaching the requested facts")


def _construct_key_delivery_plan(initial: GridState, goals: GridGoals) -> Optional[List[str]]:
    state = initial
    plan: List[str] = []
    key_goals = {key: loc for pred, key, loc in goals.atoms if pred == "key-at" and loc is not None}

    for _ in range(len(key_goals) * 4 + 8):
        if goals_hold(state, goals):
            return plan

        target_key = _choose_delivery_key(state, key_goals)
        if target_key is None:
            return None

        if state.holding != target_key:
            state = _ensure_holding_key(state, target_key, key_goals, plan)
            if state is None:
                return None

        target_loc = key_goals[target_key]
        path = _path_with_current_key(state, state.robot, target_loc)
        if path is None:
            return None
        state = _follow_path_with_unlocks(state, path, plan)
        state = _putdown(state, state.robot, target_key)
        plan.append(f"putdown {state.robot} {target_key}")

    return plan if goals_hold(state, goals) else None


def _choose_delivery_key(state: GridState, key_goals: Dict[str, str]) -> Optional[str]:
    key_at = dict(state.key_at)
    if state.holding in key_goals and key_goals[state.holding] != state.robot:
        return state.holding

    candidates = []
    for key, target in key_goals.items():
        if key_at.get(key) == target:
            continue
        loc = state.robot if state.holding == key else key_at.get(key)
        if loc is None:
            continue
        path_to_key = [state.robot] if state.holding == key else _path_with_current_key(state, state.robot, loc)
        dist = len(path_to_key) if path_to_key else 999
        candidates.append((dist, _manhattan(loc, target), key))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _ensure_holding_key(
    state: GridState,
    desired_key: str,
    key_goals: Dict[str, str],
    plan: List[str],
) -> Optional[GridState]:
    for _ in range(8):
        key_at = dict(state.key_at)
        desired_loc = key_at.get(desired_key)
        if desired_loc is None:
            return state if state.holding == desired_key else None

        path = _path_with_current_key(state, state.robot, desired_loc)
        if path is not None:
            state = _follow_path_with_unlocks(state, path, plan)
            if state.holding is None:
                state = _pickup(state, state.robot, desired_key)
                plan.append(f"pickup {state.robot} {desired_key}")
            else:
                old_key = state.holding
                state = _pickup_and_loose(state, state.robot, desired_key, old_key)
                plan.append(f"pickup-and-loose {state.robot} {desired_key} {old_key}")
            return state

        if state.holding is None:
            helper = _choose_accessible_helper_key(state, key_goals)
            if helper is None:
                return None
            helper_path = _path_open_only(state, state.robot, key_at[helper])
            if helper_path is None:
                return None
            state = _follow_path_with_unlocks(state, helper_path, plan)
            state = _pickup(state, state.robot, helper)
            plan.append(f"pickup {state.robot} {helper}")
        else:
            return None
    return None


def _choose_accessible_helper_key(state: GridState, key_goals: Dict[str, str]) -> Optional[str]:
    key_at = dict(state.key_at)
    candidates = []
    for key, loc in key_at.items():
        path = _path_open_only(state, state.robot, loc)
        if path is None:
            continue
        already_goal = key_goals.get(key) == loc
        candidates.append((already_goal, len(path), key))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _follow_path_with_unlocks(state: GridState, path: List[str], plan: List[str]) -> GridState:
    for nxt in path[1:]:
        cur = state.robot
        if nxt in state.locked:
            if state.holding is None:
                raise ValueError("cannot unlock without a key")
            shape = dict(state.key_shape)[state.holding]
            state = _unlock(state, cur, nxt, state.holding, shape)
            plan.append(f"unlock {cur} {nxt} {state.holding} {shape}")
        state = _move(state, cur, nxt)
        plan.append(f"move {cur} {nxt}")
    return state


def simulate_plan(initial: GridState, plan: List[str]) -> GridState:
    state = initial
    for action in plan:
        parts = action.split()
        name = parts[0]
        if name == "move":
            state = _move(state, parts[1], parts[2])
        elif name == "pickup":
            state = _pickup(state, parts[1], parts[2])
        elif name == "pickup-and-loose":
            state = _pickup_and_loose(state, parts[1], parts[2], parts[3])
        elif name == "putdown":
            state = _putdown(state, parts[1], parts[2])
        elif name == "unlock":
            state = _unlock(state, parts[1], parts[2], parts[3], parts[4])
        else:
            raise ValueError(f"unknown action: {action}")
    return state


def goals_hold(state: GridState, goals: GridGoals) -> bool:
    key_at = dict(state.key_at)
    for pred, arg1, arg2 in goals.atoms:
        if pred == "robot-at" and state.robot != arg1:
            return False
        if pred == "holding" and state.holding != arg1:
            return False
        if pred == "arm-empty" and state.holding is not None:
            return False
        if pred == "key-at" and key_at.get(arg1) != arg2:
            return False
        if pred == "open" and arg1 in state.locked:
            return False
        if pred == "locked" and arg1 not in state.locked:
            return False
    return True


def _successors(state: GridState, goals: GridGoals) -> Iterable[Tuple[str, GridState]]:
    key_at = dict(state.key_at)
    key_shape = dict(state.key_shape)
    lock_shape = dict(state.lock_shape)
    protected_locked = {arg1 for pred, arg1, _ in goals.atoms if pred == "locked"}

    if state.holding is None:
        for key in sorted(k for k, loc in key_at.items() if loc == state.robot):
            yield f"pickup {state.robot} {key}", _pickup(state, state.robot, key)
    else:
        old_key = state.holding
        for new_key in sorted(k for k, loc in key_at.items() if loc == state.robot):
            yield (
                f"pickup-and-loose {state.robot} {new_key} {old_key}",
                _pickup_and_loose(state, state.robot, new_key, old_key),
            )
        yield f"putdown {state.robot} {old_key}", _putdown(state, state.robot, old_key)

        shape = key_shape.get(old_key)
        if shape:
            for nxt in _neighbors(state, state.robot):
                if nxt in state.locked and nxt not in protected_locked and lock_shape.get(nxt) == shape:
                    yield f"unlock {state.robot} {nxt} {old_key} {shape}", _unlock(state, state.robot, nxt, old_key, shape)

    for nxt in _neighbors(state, state.robot):
        if nxt not in state.locked:
            yield f"move {state.robot} {nxt}", _move(state, state.robot, nxt)


def _static_impossibility(state: GridState, goals: GridGoals) -> Optional[str]:
    robot_locs = {arg1 for pred, arg1, _ in goals.atoms if pred == "robot-at"}
    if len(robot_locs) > 1:
        return "the robot can only occupy one location"

    held_keys = {arg1 for pred, arg1, _ in goals.atoms if pred == "holding"}
    if len(held_keys) > 1:
        return "the robot can hold only one key"
    if held_keys and any(pred == "arm-empty" for pred, _, _ in goals.atoms):
        return "the arm cannot be empty and holding a key"

    open_goals = {arg1 for pred, arg1, _ in goals.atoms if pred == "open"}
    locked_goals = {arg1 for pred, arg1, _ in goals.atoms if pred == "locked"}
    both = open_goals & locked_goals
    if both:
        return f"{sorted(both)[0]} cannot be both open and locked"
    for loc in locked_goals:
        if loc not in state.locked:
            return f"{loc} is not initially locked, and no action creates locked locations"

    key_goals: Dict[str, Set[str]] = {}
    for pred, key, loc in goals.atoms:
        if pred == "key-at" and loc is not None:
            key_goals.setdefault(key, set()).add(loc)
    for key, locs in key_goals.items():
        if len(locs) > 1:
            return f"{key} cannot be at multiple locations"
        if key in held_keys:
            return f"{key} cannot be held and at a grid location"

    all_keys = set(dict(state.key_shape))
    for pred, arg1, _, in goals.atoms:
        if pred in {"holding", "key-at"} and arg1 not in all_keys:
            return f"unknown key {arg1}"
    return None


def _heuristic(state: GridState, goals: GridGoals) -> int:
    key_at = dict(state.key_at)
    score = 0
    active_targets: List[str] = []
    for pred, arg1, arg2 in goals.atoms:
        if pred == "robot-at":
            active_targets.append(arg1)
        elif pred == "holding":
            if state.holding != arg1:
                loc = key_at.get(arg1, state.robot)
                active_targets.append(loc)
                score += 1
        elif pred == "key-at":
            if key_at.get(arg1) != arg2:
                if state.holding == arg1:
                    active_targets.append(arg2 or state.robot)
                else:
                    active_targets.extend([key_at.get(arg1, state.robot), arg2 or state.robot])
                score += 1
        elif not goals_hold(state, GridGoals(frozenset({(pred, arg1, arg2)}))):
            score += 1
    if active_targets:
        score += min(_manhattan(state.robot, loc) for loc in active_targets)
    score += len([loc for loc in state.locked if any(_manhattan(loc, target) <= 1 for target in active_targets)])
    return score


def _path_open_only(state: GridState, start: str, goal: str) -> Optional[List[str]]:
    return _grid_path(state, start, goal, allow_unlocks=False)


def _path_with_current_key(state: GridState, start: str, goal: str) -> Optional[List[str]]:
    return _grid_path(state, start, goal, allow_unlocks=state.holding is not None)


def _grid_path(state: GridState, start: str, goal: str, allow_unlocks: bool) -> Optional[List[str]]:
    if start == goal:
        return [start]
    held_shape = dict(state.key_shape).get(state.holding) if state.holding else None
    lock_shape = dict(state.lock_shape)
    queue: List[List[str]] = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        loc = path[-1]
        for nxt in _neighbors(state, loc):
            if nxt in seen:
                continue
            if nxt in state.locked and (not allow_unlocks or lock_shape.get(nxt) != held_shape):
                continue
            new_path = path + [nxt]
            if nxt == goal:
                return new_path
            seen.add(nxt)
            queue.append(new_path)
    return None


def _move(state: GridState, cur: str, nxt: str) -> GridState:
    if state.robot != cur or nxt in state.locked or nxt not in _neighbors(state, cur):
        raise ValueError("illegal move")
    return _replace(state, robot=nxt)


def _pickup(state: GridState, loc: str, key: str) -> GridState:
    key_at = dict(state.key_at)
    if state.robot != loc or state.holding is not None or key_at.get(key) != loc:
        raise ValueError("illegal pickup")
    del key_at[key]
    return _replace(state, holding=key, key_at=tuple(sorted(key_at.items())))


def _pickup_and_loose(state: GridState, loc: str, new_key: str, old_key: str) -> GridState:
    key_at = dict(state.key_at)
    if state.robot != loc or state.holding != old_key or key_at.get(new_key) != loc:
        raise ValueError("illegal key swap")
    del key_at[new_key]
    key_at[old_key] = loc
    return _replace(state, holding=new_key, key_at=tuple(sorted(key_at.items())))


def _putdown(state: GridState, loc: str, key: str) -> GridState:
    key_at = dict(state.key_at)
    if state.robot != loc or state.holding != key:
        raise ValueError("illegal putdown")
    key_at[key] = loc
    return _replace(state, holding=None, key_at=tuple(sorted(key_at.items())))


def _unlock(state: GridState, cur: str, lockpos: str, key: str, shape: str) -> GridState:
    key_shape = dict(state.key_shape)
    lock_shape = dict(state.lock_shape)
    if (
        state.robot != cur
        or state.holding != key
        or lockpos not in state.locked
        or lockpos not in _neighbors(state, cur)
        or key_shape.get(key) != shape
        or lock_shape.get(lockpos) != shape
    ):
        raise ValueError("illegal unlock")
    return _replace(state, locked=frozenset(set(state.locked) - {lockpos}))


def _replace(state: GridState, **changes) -> GridState:
    values = {
        "rows": state.rows,
        "cols": state.cols,
        "robot": state.robot,
        "holding": state.holding,
        "key_at": state.key_at,
        "key_shape": state.key_shape,
        "lock_shape": state.lock_shape,
        "locked": state.locked,
    }
    values.update(changes)
    return GridState(**values)


def _neighbors(state: GridState, loc: str) -> List[str]:
    r, c = _coord(loc)
    candidates = [(r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)]
    return [f"f{nr}-{nc}f" for nr, nc in candidates if 0 <= nr < state.rows and 0 <= nc < state.cols]


def _coord(loc: str) -> Tuple[int, int]:
    body = loc[1:-1]
    r, c = body.split("-")
    return int(r), int(c)


def _manhattan(a: str, b: str) -> int:
    ar, ac = _coord(a)
    br, bc = _coord(b)
    return abs(ar - br) + abs(ac - bc)
