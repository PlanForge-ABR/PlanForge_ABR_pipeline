"""Informed forward planner and simulator for floortile."""

from copy import deepcopy
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Set, Tuple

from architect.spec import Fact, FloorTileGoals, FloorTileState, SolveResult


StateKey = Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[Tuple[str, str], ...]]


def construct_plan(initial: FloorTileState, goals: FloorTileGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)
    if goals_hold(initial, goals):
        return SolveResult(True, [], "goals already hold")

    painted_goal_count = sum(1 for f in goals.facts if f[0] == "painted")
    if painted_goal_count >= 3:
        plan = _construct_painting_plan(initial, goals)
        if plan is not None:
            try:
                final_state = simulate_plan(initial, plan)
            except Exception as exc:
                return SolveResult(False, [], f"constructive painting plan failed: {exc}")
            if goals_hold(final_state, goals):
                return SolveResult(True, plan)

    plan = _astar(initial, goals, max_expansions=350000)
    if plan is None:
        return SolveResult(False, [], "search exhausted without reaching the requested state")
    try:
        final_state = simulate_plan(initial, plan)
    except Exception as exc:
        return SolveResult(False, [], f"generated plan failed during simulation: {exc}")
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "generated plan does not satisfy the requested goals")
    return SolveResult(True, plan)


def _construct_painting_plan(initial: FloorTileState, goals: FloorTileGoals) -> Optional[List[str]]:
    state = deepcopy(initial)
    plan: List[str] = []
    paint_goals = {f[1]: f[2] for f in goals.facts if f[0] == "painted"}
    coords = _grid_coords(state)

    rows = sorted({coords.get(tile, (0, 0))[0] for tile in paint_goals}, reverse=True)
    for row in rows:
        remaining = {tile for tile in paint_goals if coords.get(tile, (0, 0))[0] == row}
        while remaining:
            progressed = False
            ordered = sorted(
                remaining,
                key=lambda tile: (_constraint_score(state, tile), coords.get(tile, (0, 0))[1]),
            )
            for tile in ordered:
                attempt = _attempt_paint_tile(state, plan, tile, paint_goals[tile])
                if attempt is None:
                    continue
                state, plan = attempt
                remaining.remove(tile)
                progressed = True
                break
            if not progressed:
                return None

    if not goals_hold(state, goals):
        return None
    return plan


def _attempt_paint_tile(
    state: FloorTileState, plan: List[str], tile: str, color: str
) -> Optional[Tuple[FloorTileState, List[str]]]:
    if state.painted.get(tile) == color:
        return state, plan
    if tile not in state.clear and tile not in state.robot_at.values():
        return None
    options = _painting_stands(state, tile)
    for stand, direction in options:
        staged = _try_target_occupant_as_painter(state, plan, tile, stand, direction, color)
        if staged is not None:
            return staged

        trial = deepcopy(state)
        trial_plan = list(plan)
        robot_path = _best_robot_path(trial, stand)
        if robot_path is None:
            continue
        robot, path = robot_path
        try:
            _apply_path(trial, trial_plan, robot, path)
            occupant = next((r for r, pos in trial.robot_at.items() if pos == tile), None)
            if occupant and occupant != robot:
                if not _move_occupant_aside(trial, trial_plan, tile, {stand}):
                    continue
            if trial.robot_has[robot] != color:
                old = trial.robot_has[robot]
                trial.robot_has[robot] = color
                trial_plan.append(f"change-color {robot} {old} {color}")
            _paint(trial, robot, tile, stand, color, direction)
            trial_plan.append(f"paint-{direction} {robot} {tile} {stand} {color}")
            return trial, trial_plan
        except ValueError:
            continue
    return None


def _constraint_score(state: FloorTileState, tile: str) -> Tuple[int, int]:
    occupant = next((robot for robot, pos in state.robot_at.items() if pos == tile), None)
    if occupant is None:
        return (10, _tile_number(tile))
    exits = _clear_exit_count(state, tile, ignore_robot=occupant)
    stand_exits = 0
    for stand, _ in _painting_stands(state, tile):
        stand_occupant = next((robot for robot, pos in state.robot_at.items() if pos == stand), None)
        if stand_occupant:
            stand_exits += _clear_exit_count(state, stand, ignore_robot=stand_occupant)
    return (exits + stand_exits, _tile_number(tile))


def _clear_exit_count(state: FloorTileState, tile: str, ignore_robot: str) -> int:
    occupied = {pos for robot, pos in state.robot_at.items() if robot != ignore_robot}
    count = 0
    for nxt in state.move_edges.get(tile, {}).values():
        if nxt in state.clear and nxt not in occupied:
            count += 1
    return count


def _try_target_occupant_as_painter(
    state: FloorTileState, plan: List[str], tile: str, stand: str, direction: str, color: str
) -> Optional[Tuple[FloorTileState, List[str]]]:
    occupant = next((robot for robot, pos in state.robot_at.items() if pos == tile), None)
    if occupant is None:
        return None
    trial = deepcopy(state)
    trial_plan = list(plan)
    stand_occupant = next((robot for robot, pos in trial.robot_at.items() if pos == stand), None)
    if stand_occupant and stand_occupant != occupant:
        if not _move_occupant_aside(trial, trial_plan, stand, {tile}):
            return None
    path = _path_to(trial, occupant, stand)
    if path is None:
        return None
    try:
        _apply_path(trial, trial_plan, occupant, path)
        if trial.robot_has[occupant] != color:
            old = trial.robot_has[occupant]
            trial.robot_has[occupant] = color
            trial_plan.append(f"change-color {occupant} {old} {color}")
        _paint(trial, occupant, tile, stand, color, direction)
        trial_plan.append(f"paint-{direction} {occupant} {tile} {stand} {color}")
        return trial, trial_plan
    except ValueError:
        return None


def _move_occupant_aside(
    state: FloorTileState, plan: List[str], tile: str, forbidden: Set[str]
) -> bool:
    occupant = next((robot for robot, pos in state.robot_at.items() if pos == tile), None)
    if occupant is None:
        return True
    candidates = [t for t in state.clear if t not in forbidden]
    paths: List[Tuple[int, str, List[Tuple[str, str, str]]]] = []
    for dest in candidates:
        path = _path_to(state, occupant, dest)
        if path:
            paths.append((len(path), dest, path))
    if not paths:
        return False
    _, _, path = sorted(paths, key=lambda item: (item[0], _tile_number(item[1])))[0]
    _apply_path(state, plan, occupant, path)
    return True


def _painting_stands(state: FloorTileState, tile: str) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for stand, edges in state.move_edges.items():
        if edges.get("up") == tile:
            options.append((stand, "up"))
    for stand, edges in state.move_edges.items():
        if edges.get("down") == tile:
            options.append((stand, "down"))
    return sorted(options, key=lambda item: (_tile_number(item[0]), item[1]))


def _grid_coords(state: FloorTileState) -> Dict[str, Tuple[int, int]]:
    bottoms = [t for t in state.tiles if "down" not in state.move_edges.get(t, {})]
    bottoms = sorted(bottoms, key=_tile_number)
    coords: Dict[str, Tuple[int, int]] = {}
    for col, start in enumerate(bottoms):
        cur = start
        row = 0
        while cur and cur not in coords:
            coords[cur] = (row, col)
            cur = state.move_edges.get(cur, {}).get("up", "")
            row += 1
    for tile in state.tiles:
        coords.setdefault(tile, (0, _tile_number(tile)))
    return coords


def _best_robot_path(state: FloorTileState, dest: str) -> Optional[Tuple[str, List[Tuple[str, str, str]]]]:
    candidates: List[Tuple[int, str, List[Tuple[str, str, str]]]] = []
    for robot in state.robots:
        path = _path_to(state, robot, dest)
        if path is not None:
            candidates.append((len(path), robot, path))
    if not candidates:
        return None
    _, robot, path = sorted(candidates, key=lambda item: (item[0], _name_key(item[1])))[0]
    return robot, path


def _path_to(state: FloorTileState, robot: str, dest: str) -> Optional[List[Tuple[str, str, str]]]:
    src = state.robot_at[robot]
    if src == dest:
        return []
    occupied = {tile for other, tile in state.robot_at.items() if other != robot}
    frontier = [src]
    parent: Dict[str, Tuple[str, str]] = {}
    seen = {src}
    for tile in frontier:
        for direction in ("left", "right", "up", "down"):
            nxt = state.move_edges.get(tile, {}).get(direction)
            if not nxt or nxt in seen or nxt in occupied:
                continue
            if nxt != dest and nxt not in state.clear:
                continue
            if nxt == dest and nxt not in state.clear:
                continue
            parent[nxt] = (tile, direction)
            if nxt == dest:
                steps: List[Tuple[str, str, str]] = []
                cur = dest
                while cur != src:
                    prev, step_dir = parent[cur]
                    steps.append((step_dir, prev, cur))
                    cur = prev
                steps.reverse()
                return steps
            seen.add(nxt)
            frontier.append(nxt)
    return None


def _apply_path(state: FloorTileState, plan: List[str], robot: str, path: List[Tuple[str, str, str]]) -> None:
    for direction, src, dst in path:
        _move(state, robot, src, dst, direction)
        plan.append(f"{direction} {robot} {src} {dst}")


def validate_goal_consistency(state: FloorTileState, goals: FloorTileGoals) -> Tuple[bool, str]:
    robot_at: Dict[str, str] = {}
    tile_occupant: Dict[str, str] = {}
    robot_has: Dict[str, str] = {}
    painted: Dict[str, str] = {}
    clear: Set[str] = set()

    for fact in goals.facts:
        pred = fact[0]
        if pred == "robot-at":
            robot, tile = fact[1], fact[2]
            if robot in robot_at and robot_at[robot] != tile:
                return False, f"{robot} cannot be at two tiles"
            if tile in tile_occupant and tile_occupant[tile] != robot:
                return False, f"two robots cannot occupy {tile}"
            robot_at[robot] = tile
            tile_occupant[tile] = robot
        elif pred == "robot-has":
            robot, color = fact[1], fact[2]
            if robot in robot_has and robot_has[robot] != color:
                return False, f"{robot} cannot hold two colors"
            robot_has[robot] = color
        elif pred == "painted":
            tile, color = fact[1], fact[2]
            if tile in painted and painted[tile] != color:
                return False, f"{tile} cannot be painted two colors"
            if state.painted.get(tile) not in {None, color}:
                return False, f"{tile} is already painted {state.painted[tile]}"
            painted[tile] = color
        elif pred == "clear":
            clear.add(fact[1])

    for tile in clear:
        if tile in painted:
            return False, f"{tile} cannot be both clear and painted"
        if tile in tile_occupant:
            return False, f"{tile} cannot be both clear and occupied"
        if tile in state.painted:
            return False, f"{tile} is painted and cannot become clear"
    return True, ""


def _astar(initial: FloorTileState, goals: FloorTileGoals, max_expansions: int) -> Optional[List[str]]:
    robots = initial.robots
    colors = initial.colors or ["black", "white"]
    goal_painted = {f[1]: f[2] for f in goals.facts if f[0] == "painted"}
    useful_colors = sorted(set(colors) | set(goal_painted.values()), key=_name_key)

    start = _key(initial)
    best_g = {start: 0}
    parents: Dict[StateKey, Tuple[Optional[StateKey], Optional[str]]] = {start: (None, None)}
    heap: List[Tuple[int, int, int, StateKey]] = []
    counter = 0
    heappush(heap, (_heuristic(initial, goals), 0, counter, start))

    expansions = 0
    while heap and expansions < max_expansions:
        _, g, _, key = heappop(heap)
        if g != best_g.get(key):
            continue
        state = _from_key(initial, key)
        if goals_hold(state, goals):
            return _reconstruct(parents, key)
        expansions += 1

        for action, nxt in _successors(state, goal_painted, useful_colors):
            nkey = _key(nxt)
            ng = g + 1
            if ng >= best_g.get(nkey, 10**9):
                continue
            best_g[nkey] = ng
            parents[nkey] = (key, action)
            counter += 1
            heappush(heap, (ng + _heuristic(nxt, goals), ng, counter, nkey))
    return None


def _successors(
    state: FloorTileState, goal_painted: Dict[str, str], useful_colors: List[str]
) -> Iterable[Tuple[str, FloorTileState]]:
    for robot in sorted(state.robots, key=_name_key):
        tile = state.robot_at[robot]
        color = state.robot_has[robot]

        for direction in ("up", "down"):
            target = state.move_edges.get(tile, {}).get(direction)
            if target and target in state.clear and goal_painted.get(target) == color:
                nxt = deepcopy(state)
                nxt.clear.remove(target)
                nxt.painted[target] = color
                yield f"paint-{direction} {robot} {target} {tile} {color}", nxt

        adjacent_needed = {
            goal_painted[t]
            for d in ("up", "down")
            for t in [state.move_edges.get(tile, {}).get(d)]
            if t in state.clear and t in goal_painted and goal_painted[t] != color
        }
        color_targets = sorted(adjacent_needed or set(useful_colors), key=_name_key)
        for new_color in color_targets:
            if new_color == color:
                continue
            nxt = deepcopy(state)
            nxt.robot_has[robot] = new_color
            yield f"change-color {robot} {color} {new_color}", nxt

        for direction in ("left", "right", "up", "down"):
            target = state.move_edges.get(tile, {}).get(direction)
            if not target or target not in state.clear:
                continue
            nxt = deepcopy(state)
            nxt.robot_at[robot] = target
            nxt.clear.remove(target)
            nxt.clear.add(tile)
            yield f"{direction} {robot} {tile} {target}", nxt


def simulate_plan(initial: FloorTileState, plan: List[str]) -> FloorTileState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        name = parts[0]
        if name == "change-color":
            _, robot, old, new = parts
            if state.robot_has.get(robot) != old or new not in state.colors:
                raise ValueError(f"invalid color change: {action}")
            state.robot_has[robot] = new
        elif name in {"left", "right", "up", "down"}:
            _, robot, src, dst = parts
            _move(state, robot, src, dst, name)
        elif name in {"paint-up", "paint-down"}:
            _, robot, target, src, color = parts
            direction = name.split("-", 1)[1]
            _paint(state, robot, target, src, color, direction)
        else:
            raise ValueError(f"unknown action {action}")
    return state


def goals_hold(state: FloorTileState, goals: FloorTileGoals) -> bool:
    for fact in goals.facts:
        pred = fact[0]
        if pred == "clear" and fact[1] not in state.clear:
            return False
        if pred == "painted" and state.painted.get(fact[1]) != fact[2]:
            return False
        if pred == "robot-at" and state.robot_at.get(fact[1]) != fact[2]:
            return False
        if pred == "robot-has" and state.robot_has.get(fact[1]) != fact[2]:
            return False
    return True


def _move(state: FloorTileState, robot: str, src: str, dst: str, direction: str) -> None:
    if state.robot_at.get(robot) != src:
        raise ValueError(f"{robot} is not at {src}")
    if state.move_edges.get(src, {}).get(direction) != dst or dst not in state.clear:
        raise ValueError(f"{robot} cannot move {direction} from {src} to {dst}")
    state.robot_at[robot] = dst
    state.clear.remove(dst)
    state.clear.add(src)


def _paint(state: FloorTileState, robot: str, target: str, src: str, color: str, direction: str) -> None:
    if state.robot_at.get(robot) != src or state.robot_has.get(robot) != color:
        raise ValueError(f"{robot} cannot paint from {src} with {color}")
    if state.move_edges.get(src, {}).get(direction) != target or target not in state.clear:
        raise ValueError(f"{target} is not paintable from {src}")
    state.clear.remove(target)
    state.painted[target] = color


def _heuristic(state: FloorTileState, goals: FloorTileGoals) -> int:
    total = 0
    distances = _all_pairs_distances(state)
    for fact in goals.facts:
        pred = fact[0]
        if pred == "painted":
            tile, color = fact[1], fact[2]
            if state.painted.get(tile) == color:
                continue
            best = 20
            for robot in state.robots:
                pos = state.robot_at[robot]
                color_cost = 0 if state.robot_has[robot] == color else 1
                stands = [
                    src
                    for src, edges in state.move_edges.items()
                    if edges.get("up") == tile or edges.get("down") == tile
                ]
                dist = min((distances.get((pos, s), 20) for s in stands), default=20)
                best = min(best, dist + color_cost + 1)
            total += best * 3
        elif pred == "robot-at":
            total += min(_all_pairs_distances(state).get((state.robot_at.get(fact[1], ""), fact[2]), 20), 20)
        elif pred == "robot-has" and state.robot_has.get(fact[1]) != fact[2]:
            total += 1
        elif pred == "clear" and fact[1] not in state.clear:
            total += 2
    return total


def _all_pairs_distances(state: FloorTileState) -> Dict[Tuple[str, str], int]:
    dist: Dict[Tuple[str, str], int] = {}
    for start in state.tiles:
        frontier = [(start, 0)]
        seen = {start}
        for tile, d in frontier:
            dist[(start, tile)] = d
            for nxt in state.move_edges.get(tile, {}).values():
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, d + 1))
    return dist


def _key(state: FloorTileState) -> StateKey:
    return (
        tuple(state.robot_at[r] for r in state.robots),
        tuple(state.robot_has[r] for r in state.robots),
        tuple(sorted(state.clear, key=_name_key)),
        tuple(sorted(state.painted.items(), key=lambda item: _name_key(item[0]))),
    )


def _from_key(template: FloorTileState, key: StateKey) -> FloorTileState:
    positions, colors, clear, painted = key
    return FloorTileState(
        colors=template.colors,
        robots=template.robots,
        tiles=template.tiles,
        move_edges=template.move_edges,
        robot_at=dict(zip(template.robots, positions)),
        robot_has=dict(zip(template.robots, colors)),
        clear=set(clear),
        painted=dict(painted),
    )


def _reconstruct(parents: Dict[StateKey, Tuple[Optional[StateKey], Optional[str]]], key: StateKey) -> List[str]:
    actions: List[str] = []
    while True:
        parent, action = parents[key]
        if parent is None:
            break
        actions.append(action or "")
        key = parent
    actions.reverse()
    return actions


def _name_key(name: str):
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return (prefix, int(digits) if digits else -1, name)


def _tile_number(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1
