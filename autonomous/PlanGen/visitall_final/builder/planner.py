"""Builder implementation of the architect's visitall methods."""

from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from architect.spec import Location, SolveResult, VisitAllGoals, VisitAllState


def construct_plan(initial: VisitAllState, goals: VisitAllGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)

    state = deepcopy(initial)
    plan: List[str] = []
    pending = set(goals.visited - state.visited)

    while pending:
        target, path = _nearest_pending_path(state.current, pending, state.connected)
        if target is None or path is None:
            return SolveResult(False, [], "a required visited location is unreachable")
        _apply_path(state, plan, path)
        pending -= state.visited

    if goals.at and state.current != goals.at:
        path = _shortest_path(state.current, goals.at, state.connected)
        if path is None:
            return SolveResult(False, [], "the required final robot location is unreachable")
        _apply_path(state, plan, path)

    final_state = simulate_plan(initial, plan)
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "constructed route did not satisfy all requested facts")
    return SolveResult(True, plan)


def validate_goal_consistency(state: VisitAllState, goals: VisitAllGoals) -> Tuple[bool, str]:
    if goals.at_conflict:
        return False, "the robot cannot be at two different places at once"
    if state.current not in state.locations:
        return False, "initial robot location is unavailable"

    all_goal_locs = set(goals.visited)
    if goals.at:
        all_goal_locs.add(goals.at)

    unavailable_goals = sorted(loc for loc in all_goal_locs if loc not in state.locations)
    if unavailable_goals:
        return False, f"goal references unavailable or out-of-grid locations: {', '.join(unavailable_goals)}"

    reachable = _reachable_from(state.current, state.connected)
    unreachable = sorted(loc for loc in all_goal_locs if loc not in reachable)
    if unreachable:
        return False, f"goal locations are disconnected from the robot: {', '.join(unreachable)}"
    return True, ""


def simulate_plan(initial: VisitAllState, plan: List[str]) -> VisitAllState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        if len(parts) != 3 or parts[0] != "move":
            raise ValueError(f"unknown action: {action}")
        _, src, dst = parts
        if state.current != src:
            raise ValueError(f"move source {src} does not match current location {state.current}")
        if dst not in state.connected.get(src, set()):
            raise ValueError(f"{src} is not connected to {dst}")
        state.current = dst
        state.visited.add(dst)
    return state


def goals_hold(state: VisitAllState, goals: VisitAllGoals) -> bool:
    if goals.at and state.current != goals.at:
        return False
    return goals.visited.issubset(state.visited)


def _nearest_pending_path(
    start: Location,
    pending: Set[Location],
    connected: Dict[Location, Set[Location]],
) -> Tuple[Optional[Location], Optional[List[Location]]]:
    queue = deque([start])
    parent: Dict[Location, Optional[Location]] = {start: None}

    while queue:
        loc = queue.popleft()
        if loc in pending:
            return loc, _reconstruct_path(loc, parent)
        for nxt in _ordered_neighbors(loc, connected):
            if nxt not in parent:
                parent[nxt] = loc
                queue.append(nxt)
    return None, None


def _shortest_path(
    start: Location,
    goal: Location,
    connected: Dict[Location, Set[Location]],
) -> Optional[List[Location]]:
    queue = deque([start])
    parent: Dict[Location, Optional[Location]] = {start: None}

    while queue:
        loc = queue.popleft()
        if loc == goal:
            return _reconstruct_path(loc, parent)
        for nxt in _ordered_neighbors(loc, connected):
            if nxt not in parent:
                parent[nxt] = loc
                queue.append(nxt)
    return None


def _apply_path(state: VisitAllState, plan: List[str], path: List[Location]) -> None:
    for src, dst in zip(path, path[1:]):
        if dst not in state.connected.get(src, set()):
            raise ValueError(f"{src} is not connected to {dst}")
        plan.append(f"move {src} {dst}")
        state.current = dst
        state.visited.add(dst)


def _reachable_from(start: Location, connected: Dict[Location, Set[Location]]) -> Set[Location]:
    seen = {start}
    queue = deque([start])
    while queue:
        loc = queue.popleft()
        for nxt in connected.get(loc, set()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _reconstruct_path(goal: Location, parent: Dict[Location, Optional[Location]]) -> List[Location]:
    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _ordered_neighbors(loc: Location, connected: Dict[Location, Set[Location]]) -> List[Location]:
    return sorted(connected.get(loc, set()), key=_loc_key)


def _loc_key(loc: Location) -> Tuple[int, int]:
    x_part, y_part = loc.split("-")[1:]
    return int(x_part[1:]), int(y_part[1:])
