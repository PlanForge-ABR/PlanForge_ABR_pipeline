"""Builder implementation of frogs_jumping planner methods."""

from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from architect.spec import FrogGoals, FrogState, SolveResult, position_map


Board = Tuple[str, ...]


def construct_plan(initial: FrogState, goals: FrogGoals) -> SolveResult:
    impossible = _static_impossibility(initial, goals)
    if impossible:
        return SolveResult(False, tuple(), impossible)
    if goals_hold(initial, goals):
        return SolveResult(True, tuple(), "goals already hold")

    result = _standard_path_continuation(initial.board, goals)
    if result is not None:
        return SolveResult(True, tuple(result), "goals reached on the standard frogs-jumping path")

    result = _canonical_forward(initial.board, goals, max_steps=_remaining_move_bound(initial.board))
    if result is not None:
        return SolveResult(True, tuple(result), "goals reached by canonical monotone sequence")

    if len(initial.board) <= 41:
        result = _best_first(initial.board, goals, max_expansions=250000)
        if result is not None:
            return SolveResult(True, tuple(result), "goals reached by bounded best-first search")

    return SolveResult(False, tuple(), "no legal monotone continuation reached the requested facts")


def goals_hold(state: FrogState, goals: FrogGoals) -> bool:
    return _goals_hold_board(state.board, goals)


def simulate_plan(initial: FrogState, plan: Iterable[str]) -> FrogState:
    board = list(initial.board)
    for action in plan:
        parts = action.split()
        if len(parts) not in (4, 6):
            raise ValueError(f"unknown action: {action}")
        kind, frog = parts[0], parts[1]
        from_pad = _pad_num(parts[2])
        to_pad = _pad_num(parts[3] if kind.startswith("slide") else parts[4])
        _apply_checked(board, kind, frog, from_pad, to_pad, action)
    return _state_from_board(initial, tuple(board))


def legal_moves(board: Board) -> List[Tuple[str, Board]]:
    moves = []
    empty = board.index("_") + 1
    n = len(board)

    candidates = (
        (empty - 2, empty - 1, "jump-right"),
        (empty + 2, empty + 1, "jump-left"),
        (empty + 1, None, "slide-left"),
        (empty - 1, None, "slide-right"),
    )
    for src, mid, kind in candidates:
        if src < 1 or src > n:
            continue
        frog = board[src - 1]
        if frog == "_":
            continue
        if kind.endswith("right") and not frog.startswith("l"):
            continue
        if kind.endswith("left") and not frog.startswith("r"):
            continue
        if mid is not None and not _can_jump_over(frog, board[mid - 1]):
            continue

        next_board = list(board)
        next_board[empty - 1] = frog
        next_board[src - 1] = "_"
        if mid is None:
            action = f"{kind} {frog} p{src} p{empty}"
        else:
            jumped = board[mid - 1]
            action = f"{kind} {frog} p{src} p{mid} p{empty} {jumped}"
        moves.append((action, tuple(next_board)))
    return moves


def _standard_path_continuation(board: Board, goals: FrogGoals) -> Optional[List[str]]:
    left = sorted((frog for frog in board if frog.startswith("l")), key=_frog_key)
    right = sorted((frog for frog in board if frog.startswith("r")), key=_frog_key)
    if len(left) != len(right):
        return None

    start: Board = tuple(left + ["_"] + right)
    current = start
    states: List[Board] = [current]
    actions: List[str] = []
    seen = {current}

    n = len(left)
    group_lengths = list(range(1, n + 1)) + [n, n] + list(range(n - 1, 0, -1))
    direction = "left"
    for group_len in group_lengths:
        for _ in range(group_len):
            move = _directed_move(current, direction)
            if move is None:
                return None
            action, current = move
            actions.append(action)
            states.append(current)
            if current in seen:
                return None
            seen.add(current)
        direction = "right" if direction == "left" else "left"

    try:
        start_index = states.index(board)
    except ValueError:
        return None

    plan: List[str] = []
    for idx in range(start_index, len(actions)):
        plan.append(actions[idx])
        if _goals_hold_board(states[idx + 1], goals):
            return plan
    return None


def _directed_move(board: Board, direction: str) -> Optional[Tuple[str, Board]]:
    empty = board.index("_") + 1
    if direction == "left":
        jump_src = empty + 2
        if jump_src <= len(board) and board[jump_src - 1].startswith("r") and board[empty].startswith("l"):
            frog = board[jump_src - 1]
            jumped = board[empty]
            return _move_result(board, f"jump-left {frog} p{jump_src} p{empty + 1} p{empty} {jumped}", jump_src, empty)
        slide_src = empty + 1
        if slide_src <= len(board) and board[slide_src - 1].startswith("r"):
            frog = board[slide_src - 1]
            return _move_result(board, f"slide-left {frog} p{slide_src} p{empty}", slide_src, empty)
    else:
        jump_src = empty - 2
        if jump_src >= 1 and board[jump_src - 1].startswith("l") and board[empty - 2].startswith("r"):
            frog = board[jump_src - 1]
            jumped = board[empty - 2]
            return _move_result(board, f"jump-right {frog} p{jump_src} p{empty - 1} p{empty} {jumped}", jump_src, empty)
        slide_src = empty - 1
        if slide_src >= 1 and board[slide_src - 1].startswith("l"):
            frog = board[slide_src - 1]
            return _move_result(board, f"slide-right {frog} p{slide_src} p{empty}", slide_src, empty)
    return None


def _move_result(board: Board, action: str, src: int, dst: int) -> Tuple[str, Board]:
    next_board = list(board)
    next_board[dst - 1] = next_board[src - 1]
    next_board[src - 1] = "_"
    return action, tuple(next_board)


def _canonical_forward(board: Board, goals: FrogGoals, max_steps: int) -> Optional[List[str]]:
    current = board
    plan: List[str] = []
    seen = {current}
    for _ in range(max_steps + 1):
        ordered = _ordered_moves(current, goals)
        if not ordered:
            return None
        action, current = ordered[0]
        plan.append(action)
        if _goals_hold_board(current, goals):
            return plan
        if current in seen:
            return None
        seen.add(current)
    return None


def _ordered_moves(board: Board, goals: FrogGoals) -> List[Tuple[str, Board]]:
    priority = {"jump-right": 0, "jump-left": 1, "slide-left": 2, "slide-right": 3}
    return sorted(
        legal_moves(board),
        key=lambda item: (_heuristic(item[1], goals), priority[item[0].split()[0]]),
    )


def _best_first(board: Board, goals: FrogGoals, max_expansions: int) -> Optional[List[str]]:
    queue = []
    counter = 0
    heappush(queue, (_heuristic(board, goals), 0, counter, board, ()))
    best_depth: Dict[Board, int] = {board: 0}

    while queue and len(best_depth) <= max_expansions:
        _, depth, _, current, path = heappop(queue)
        if _goals_hold_board(current, goals):
            return list(path)
        if depth != best_depth[current]:
            continue
        for action, nxt in legal_moves(current):
            next_depth = depth + 1
            if next_depth >= best_depth.get(nxt, 10**9):
                continue
            best_depth[nxt] = next_depth
            counter += 1
            heappush(
                queue,
                (next_depth + _heuristic(nxt, goals), next_depth, counter, nxt, path + (action,)),
            )
    return None


def _heuristic(board: Board, goals: FrogGoals) -> int:
    positions = {frog: idx + 1 for idx, frog in enumerate(board) if frog != "_"}
    cost = 0
    for frog, target in goals.at:
        pos = positions.get(frog)
        if pos is not None:
            cost += abs(pos - target)
    for pad in goals.empty:
        cost += 0 if board[pad - 1] == "_" else 1
    return cost


def _static_impossibility(state: FrogState, goals: FrogGoals) -> Optional[str]:
    positions = position_map(state)
    n = len(state.board)
    frog_targets: Dict[str, set] = {}
    pad_targets: Dict[int, set] = {}

    for frog, pad in goals.at:
        if frog not in positions:
            return f"unknown frog {frog}"
        if pad < 1 or pad > n:
            return f"unknown lily pad p{pad}"
        frog_targets.setdefault(frog, set()).add(pad)
        pad_targets.setdefault(pad, set()).add(frog)
        if frog.startswith("l") and pad < positions[frog]:
            return f"{frog} cannot move left from p{positions[frog]} to p{pad}"
        if frog.startswith("r") and pad > positions[frog]:
            return f"{frog} cannot move right from p{positions[frog]} to p{pad}"

    for pad in goals.empty:
        if pad < 1 or pad > n:
            return f"unknown lily pad p{pad}"
        if pad in pad_targets:
            return f"p{pad} cannot be empty and occupied"

    for frog, pads in frog_targets.items():
        if len(pads) > 1:
            return f"{frog} cannot be at multiple lily pads"
    for pad, frogs in pad_targets.items():
        if len(frogs) > 1:
            return f"multiple frogs cannot occupy p{pad}"

    ordered_goals = sorted(goals.at, key=lambda item: positions[item[0]])
    for (frog_a, target_a), (frog_b, target_b) in zip(ordered_goals, ordered_goals[1:]):
        if frog_a[0] == frog_b[0] and target_a >= target_b:
            return "same-direction frogs cannot overtake each other"
    return None


def _remaining_move_bound(board: Board) -> int:
    left = sum(1 for frog in board if frog.startswith("l"))
    right = sum(1 for frog in board if frog.startswith("r"))
    return left * right + left + right


def _goals_hold_board(board: Sequence[str], goals: FrogGoals) -> bool:
    for frog, pad in goals.at:
        if pad < 1 or pad > len(board) or board[pad - 1] != frog:
            return False
    return all(1 <= pad <= len(board) and board[pad - 1] == "_" for pad in goals.empty)


def _apply_checked(board: List[str], kind: str, frog: str, from_pad: int, to_pad: int, action: str) -> None:
    if board[from_pad - 1] != frog or board[to_pad - 1] != "_":
        raise ValueError(f"precondition failed: {action}")
    if kind == "slide-left":
        ok = frog.startswith("r") and to_pad == from_pad - 1
    elif kind == "slide-right":
        ok = frog.startswith("l") and to_pad == from_pad + 1
    elif kind == "jump-left":
        ok = frog.startswith("r") and to_pad == from_pad - 2 and board[from_pad - 2].startswith("l")
    elif kind == "jump-right":
        ok = frog.startswith("l") and to_pad == from_pad + 2 and board[from_pad].startswith("r")
    else:
        ok = False
    if not ok:
        raise ValueError(f"illegal action: {action}")
    board[to_pad - 1] = frog
    board[from_pad - 1] = "_"


def _state_from_board(template: FrogState, board: Board) -> FrogState:
    positions = tuple(sorted((frog, idx + 1) for idx, frog in enumerate(board) if frog != "_"))
    return FrogState(board=board, positions=positions, left_frogs=template.left_frogs, right_frogs=template.right_frogs)


def _pad_num(token: str) -> int:
    if not token.startswith("p"):
        raise ValueError(f"expected lily pad token, got {token}")
    return int(token[1:])


def _can_jump_over(frog: str, middle: str) -> bool:
    return (frog.startswith("l") and middle.startswith("r")) or (
        frog.startswith("r") and middle.startswith("l")
    )


def _frog_key(name: str) -> int:
    return int(name[1:])
