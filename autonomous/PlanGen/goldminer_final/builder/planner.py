"""Builder implementation of the architect's goldminer methods."""

from heapq import heappop, heappush
from itertools import count
from typing import Dict, Iterable, List, Optional, Set, Tuple

from architect.spec import GMGoals, GMState, SolveResult


SEARCH_LIMIT = 250000


def construct_plan(initial: GMState, goals: GMGoals) -> SolveResult:
    impossible = _static_impossibility(initial, goals)
    if impossible:
        return SolveResult(False, [], impossible)
    if goals_hold(initial, goals):
        return SolveResult(True, [])

    if any(pred == "holds-gold" for pred, _ in goals.atoms):
        plan = _construct_gold_plan(initial, goals)
        if plan is not None:
            return SolveResult(True, plan)

    queue: List[Tuple[int, int, int, GMState, List[str]]] = []
    serial = count()
    heappush(queue, (_heuristic(initial, goals), 0, next(serial), initial, []))
    best_depth: Dict[GMState, int] = {initial: 0}
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
            priority = new_depth + _heuristic(nxt, goals)
            heappush(queue, (priority, new_depth, next(serial), nxt, new_plan))

    return SolveResult(False, [], "search exhausted without reaching the requested facts")


def _construct_gold_plan(initial: GMState, goals: GMGoals) -> Optional[List[str]]:
    for gold_loc in sorted(initial.gold, key=lambda loc: (_manhattan(initial.robot, loc), loc)):
        attempt = _try_gold_location(initial, goals, gold_loc)
        if attempt is not None:
            return attempt
    return None


def _try_gold_location(initial: GMState, goals: GMGoals, gold_loc: str) -> Optional[List[str]]:
    state = initial
    plan: List[str] = []

    if state.holding == "gold":
        return _finish_after_gold(state, goals, plan)
    if state.holding not in {"empty", "laser"}:
        return None

    if gold_loc in state.clear:
        path = _clear_path(state, state.robot, gold_loc)
        if path is not None:
            state = _move_along_clear_path(state, path, plan)
            if state.holding == "empty":
                state = _pick_gold(state, gold_loc)
                plan.append(f"pick-gold {gold_loc}")
                return _finish_after_gold(state, goals, plan)

    if state.holding != "laser":
        if state.laser_at is None:
            return None
        path_to_laser = _clear_path(state, state.robot, state.laser_at)
        if path_to_laser is None:
            return None
        state = _move_along_clear_path(state, path_to_laser, plan)
        state = _pickup_laser(state, state.robot)
        plan.append(f"pickup-laser {state.robot}")

    approaches = [n for n in _neighbors(state, gold_loc) if n != gold_loc]
    approaches.sort(key=lambda loc: (_manhattan(state.robot, loc), loc))
    for approach in approaches:
        trial_state = state
        trial_plan = list(plan)
        carve_path = _grid_path(trial_state, trial_state.robot, approach, forbidden={gold_loc})
        if carve_path is None:
            continue
        try:
            trial_state = _carve_path_with_laser(trial_state, carve_path, trial_plan)
            if gold_loc in trial_state.clear:
                move_path = _clear_path(trial_state, trial_state.robot, gold_loc)
                if move_path is not None:
                    trial_state = _move_along_clear_path(trial_state, move_path, trial_plan)
                    trial_state = _pick_gold(trial_state, gold_loc)
                    trial_plan.append(f"pick-gold {gold_loc}")
                    return _finish_after_gold(trial_state, goals, trial_plan)

            trial_state = _putdown_laser(trial_state, trial_state.robot)
            trial_plan.append(f"putdown-laser {trial_state.robot}")

            path_to_bomb = _clear_path(trial_state, trial_state.robot, trial_state.bomb_at)
            if path_to_bomb is None:
                continue
            trial_state = _move_along_clear_path(trial_state, path_to_bomb, trial_plan)
            trial_state = _pickup_bomb(trial_state, trial_state.robot)
            trial_plan.append(f"pickup-bomb {trial_state.robot}")

            path_back = _clear_path(trial_state, trial_state.robot, approach)
            if path_back is None:
                continue
            trial_state = _move_along_clear_path(trial_state, path_back, trial_plan)
            if gold_loc not in trial_state.soft or gold_loc not in _neighbors(trial_state, trial_state.robot):
                continue
            trial_state = _detonate_bomb(trial_state, trial_state.robot, gold_loc)
            trial_plan.append(f"detonate-bomb {approach} {gold_loc}")
            trial_state = _move(trial_state, approach, gold_loc)
            trial_plan.append(f"move {approach} {gold_loc}")
            trial_state = _pick_gold(trial_state, gold_loc)
            trial_plan.append(f"pick-gold {gold_loc}")
            return _finish_after_gold(trial_state, goals, trial_plan)
        except ValueError:
            continue
    return None


def _finish_after_gold(state: GMState, goals: GMGoals, plan: List[str]) -> Optional[List[str]]:
    for pred, loc in goals.atoms:
        if pred == "robot-at" and loc and state.robot != loc:
            path = _clear_path(state, state.robot, loc)
            if path is None:
                return None
            state = _move_along_clear_path(state, path, plan)
    if goals_hold(state, goals):
        return plan
    return None


def _carve_path_with_laser(state: GMState, path: List[str], plan: List[str]) -> GMState:
    for y in path[1:]:
        x = state.robot
        if y not in state.clear:
            state = _fire_laser(state, x, y)
            plan.append(f"fire-laser {x} {y}")
        state = _move(state, x, y)
        plan.append(f"move {x} {y}")
    return state


def _move_along_clear_path(state: GMState, path: List[str], plan: List[str]) -> GMState:
    for y in path[1:]:
        x = state.robot
        state = _move(state, x, y)
        plan.append(f"move {x} {y}")
    return state


def _clear_path(state: GMState, start: str, goal: str) -> Optional[List[str]]:
    return _grid_path(state, start, goal, forbidden=set(), clear_only=True)


def _grid_path(
    state: GMState,
    start: str,
    goal: str,
    forbidden: Set[str],
    clear_only: bool = False,
) -> Optional[List[str]]:
    if start in forbidden or goal in forbidden:
        return None
    queue: List[List[str]] = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        loc = path[-1]
        if loc == goal:
            return path
        for nxt in _neighbors(state, loc):
            if nxt in seen or nxt in forbidden:
                continue
            if clear_only and nxt not in state.clear:
                continue
            seen.add(nxt)
            queue.append(path + [nxt])
    return None


def simulate_plan(initial: GMState, plan: List[str]) -> GMState:
    state = initial
    for action in plan:
        parts = action.split()
        name = parts[0]
        if name == "move":
            state = _move(state, parts[1], parts[2])
        elif name == "fire-laser":
            state = _fire_laser(state, parts[1], parts[2])
        elif name == "pickup-laser":
            state = _pickup_laser(state, parts[1])
        elif name == "putdown-laser":
            state = _putdown_laser(state, parts[1])
        elif name == "pickup-bomb":
            state = _pickup_bomb(state, parts[1])
        elif name == "detonate-bomb":
            state = _detonate_bomb(state, parts[1], parts[2])
        elif name == "pick-gold":
            state = _pick_gold(state, parts[1])
        else:
            raise ValueError(f"unknown action: {action}")
    return state


def goals_hold(state: GMState, goals: GMGoals) -> bool:
    for pred, loc in goals.atoms:
        if pred == "robot-at" and state.robot != loc:
            return False
        if pred == "holds-gold" and state.holding != "gold":
            return False
        if pred == "holds-laser" and state.holding != "laser":
            return False
        if pred == "holds-bomb" and state.holding != "bomb":
            return False
        if pred == "arm-empty" and state.holding != "empty":
            return False
        if pred == "soft-rock-at" and loc not in state.soft:
            return False
        if pred == "hard-rock-at" and loc not in state.hard:
            return False
        if pred == "gold-at" and loc not in state.gold:
            return False
        if pred == "laser-at" and state.laser_at != loc:
            return False
        if pred == "clear" and loc not in state.clear:
            return False
    return True


def _successors(state: GMState, goals: GMGoals) -> Iterable[Tuple[str, GMState]]:
    x = state.robot

    if state.holding == "empty":
        if x in state.gold:
            yield f"pick-gold {x}", _pick_gold(state, x)
        if state.laser_at == x:
            yield f"pickup-laser {x}", _pickup_laser(state, x)
        if state.bomb_at == x:
            yield f"pickup-bomb {x}", _pickup_bomb(state, x)

    if state.holding == "laser":
        yield f"putdown-laser {x}", _putdown_laser(state, x)
        protected = _protected_locations(goals)
        for y in _neighbors(state, x):
            if y not in state.clear and y not in protected:
                yield f"fire-laser {x} {y}", _fire_laser(state, x, y)

    if state.holding == "bomb":
        protected_soft = {loc for pred, loc in goals.atoms if pred == "soft-rock-at" and loc}
        for y in _neighbors(state, x):
            if y in state.soft and y not in protected_soft:
                yield f"detonate-bomb {x} {y}", _detonate_bomb(state, x, y)

    for y in _neighbors(state, x):
        if y in state.clear:
            yield f"move {x} {y}", _move(state, x, y)


def _static_impossibility(state: GMState, goals: GMGoals) -> Optional[str]:
    seen_holding: Set[str] = set()
    for pred, loc in goals.atoms:
        if pred == "soft-rock-at" and loc not in state.soft:
            return f"{loc} is not initially soft rock, and no action creates soft rock"
        if pred == "hard-rock-at" and loc not in state.hard:
            return f"{loc} is not initially hard rock, and no action creates hard rock"
        if pred == "gold-at" and loc not in state.gold:
            return f"{loc} is not initially gold, and no action creates gold"
        if pred.startswith("holds-"):
            seen_holding.add(pred)
    if len(seen_holding) > 1:
        return "the arm can hold only one item"
    if "arm-empty" in {pred for pred, _ in goals.atoms} and seen_holding:
        return "the arm cannot be empty and holding an item"
    return None


def _heuristic(state: GMState, goals: GMGoals) -> int:
    score = 0
    for pred, loc in goals.atoms:
        if pred == "robot-at" and loc:
            score += _manhattan(state.robot, loc)
        elif pred == "holds-gold":
            score += 0 if state.holding == "gold" else min((_manhattan(state.robot, g) for g in state.gold), default=20) + 1
        elif pred == "holds-laser":
            score += 0 if state.holding == "laser" else (0 if state.laser_at is None else _manhattan(state.robot, state.laser_at) + 1)
        elif pred == "holds-bomb":
            score += 0 if state.holding == "bomb" else _manhattan(state.robot, state.bomb_at) + 1
        elif not goals_hold(state, GMGoals(frozenset({(pred, loc)}))):
            score += 4
    if state.holding == "laser":
        score -= 1
    return max(score, 0)


def _protected_locations(goals: GMGoals) -> Set[str]:
    protected: Set[str] = set()
    for pred, loc in goals.atoms:
        if pred in {"soft-rock-at", "hard-rock-at", "gold-at"} and loc:
            protected.add(loc)
    return protected


def _neighbors(state: GMState, loc: str) -> List[str]:
    r, c = _coord(loc)
    candidates = [(r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)]
    return [f"f{nr}-{nc}f" for nr, nc in candidates if 0 <= nr < state.rows and 0 <= nc < state.cols]


def _move(state: GMState, x: str, y: str) -> GMState:
    if state.robot != x or y not in state.clear or y not in _neighbors(state, x):
        raise ValueError("illegal move")
    return _replace(state, robot=y)


def _fire_laser(state: GMState, x: str, y: str) -> GMState:
    if state.robot != x or state.holding != "laser" or y not in _neighbors(state, x):
        raise ValueError("illegal laser action")
    return _replace(
        state,
        clear=frozenset(set(state.clear) | {y}),
        soft=frozenset(set(state.soft) - {y}),
        hard=frozenset(set(state.hard) - {y}),
        gold=frozenset(set(state.gold) - {y}),
    )


def _pickup_laser(state: GMState, x: str) -> GMState:
    if state.robot != x or state.holding != "empty" or state.laser_at != x:
        raise ValueError("illegal laser pickup")
    return _replace(state, holding="laser", laser_at=None)


def _putdown_laser(state: GMState, x: str) -> GMState:
    if state.robot != x or state.holding != "laser":
        raise ValueError("illegal laser putdown")
    return _replace(state, holding="empty", laser_at=x)


def _pickup_bomb(state: GMState, x: str) -> GMState:
    if state.robot != x or state.holding != "empty" or state.bomb_at != x:
        raise ValueError("illegal bomb pickup")
    return _replace(state, holding="bomb")


def _detonate_bomb(state: GMState, x: str, y: str) -> GMState:
    if state.robot != x or state.holding != "bomb" or y not in state.soft or y not in _neighbors(state, x):
        raise ValueError("illegal bomb detonation")
    return _replace(
        state,
        holding="empty",
        clear=frozenset(set(state.clear) | {y}),
        soft=frozenset(set(state.soft) - {y}),
    )


def _pick_gold(state: GMState, x: str) -> GMState:
    if state.robot != x or state.holding != "empty" or x not in state.gold:
        raise ValueError("illegal gold pickup")
    return _replace(state, holding="gold")


def _replace(state: GMState, **changes) -> GMState:
    values = {
        "rows": state.rows,
        "cols": state.cols,
        "robot": state.robot,
        "holding": state.holding,
        "bomb_at": state.bomb_at,
        "laser_at": state.laser_at,
        "clear": state.clear,
        "soft": state.soft,
        "hard": state.hard,
        "gold": state.gold,
    }
    values.update(changes)
    return GMState(**values)


def _coord(loc: str) -> Tuple[int, int]:
    body = loc[1:-1]
    r, c = body.split("-")
    return int(r), int(c)


def _manhattan(a: str, b: str) -> int:
    ar, ac = _coord(a)
    br, bc = _coord(b)
    return abs(ar - br) + abs(ac - bc)
