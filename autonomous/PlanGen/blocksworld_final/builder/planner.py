"""Builder implementation of the architect's blocksworld methods."""

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from architect.spec import BWGoals, BWState, SolveResult


def construct_plan(initial: BWState, goals: BWGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)

    state = deepcopy(initial)
    plan: List[str] = []

    if state.holding is not None:
        _put_down(state, plan, state.holding)

    _clear_world_to_table(state, plan)
    _build_goal_stacks(state, goals.on, plan)

    if goals.holding:
        held = goals.holding
        if state.holding and state.holding != held:
            _put_down(state, plan, state.holding)
        if state.holding is None:
            if held in state.on:
                _unstack(state, plan, held, state.on[held])
            elif held in state.ontable:
                _pick_up(state, plan, held)

    if goals.handempty and state.holding is not None:
        _put_down(state, plan, state.holding)

    final_state = simulate_plan(initial, plan)
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "constructive planner could not satisfy the requested facts")
    return SolveResult(True, plan)


def validate_goal_consistency(state: BWState, goals: BWGoals) -> Tuple[bool, str]:
    support_of: Dict[str, str] = {}
    top_of: Dict[str, str] = {}

    for child, parent in goals.on:
        if child == parent:
            return False, f"{child} cannot be on itself"
        if child in support_of and support_of[child] != parent:
            return False, f"{child} cannot be on two different supports"
        if parent in top_of and top_of[parent] != child:
            return False, f"two blocks cannot both be directly on {parent}"
        support_of[child] = parent
        top_of[parent] = child

    if _has_cycle(support_of):
        return False, "goal on-relations contain a cycle"

    if goals.holding_conflict:
        return False, "the arm cannot hold two different blocks at the same time"

    if goals.holding and goals.handempty:
        return False, "the arm cannot be empty while holding a block"

    if goals.holding:
        if goals.holding in support_of:
            return False, f"{goals.holding} cannot be held and also on another object"
        if goals.holding in goals.ontable:
            return False, f"{goals.holding} cannot be held and on the table"
        if goals.holding in top_of:
            return False, f"{goals.holding} cannot be held with another block on top"

    for block in goals.clear:
        if block in top_of:
            return False, f"{block} cannot be clear with {top_of[block]} on top"

    for block in goals.ontable:
        if block in support_of:
            return False, f"{block} cannot be on the table and on another object"

    return True, ""


def _has_cycle(support_of: Dict[str, str]) -> bool:
    for start in support_of:
        seen = set()
        cur = start
        while cur in support_of:
            if cur in seen:
                return True
            seen.add(cur)
            cur = support_of[cur]
    return False


def _clear_world_to_table(state: BWState, plan: List[str]) -> None:
    while state.on:
        clear_blocks = _clear_blocks(state)
        movable = sorted([b for b in state.on if b in clear_blocks], key=_block_key)
        if not movable:
            raise RuntimeError("invalid blocksworld state: no clear block can be unstacked")
        block = movable[0]
        parent = state.on[block]
        _unstack(state, plan, block, parent)
        _put_down(state, plan, block)


def _build_goal_stacks(state: BWState, goal_on: Set[Tuple[str, str]], plan: List[str]) -> None:
    support_of = {child: parent for child, parent in goal_on}
    top_of = {parent: child for child, parent in goal_on}
    bases = sorted({parent for _, parent in goal_on if parent not in support_of}, key=_block_key)

    for base in bases:
        parent = base
        while parent in top_of:
            child = top_of[parent]
            if state.holding is not None:
                _put_down(state, plan, state.holding)
            if child not in state.ontable:
                raise RuntimeError(f"{child} was expected to be available on the table")
            _pick_up(state, plan, child)
            _stack(state, plan, child, parent)
            parent = child


def simulate_plan(initial: BWState, plan: List[str]) -> BWState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        name = parts[0]
        if name == "pick-up":
            _pick_up(state, None, parts[1])
        elif name == "put-down":
            _put_down(state, None, parts[1])
        elif name == "stack":
            _stack(state, None, parts[1], parts[2])
        elif name == "unstack":
            _unstack(state, None, parts[1], parts[2])
        else:
            raise ValueError(f"unknown action: {action}")
    return state


def goals_hold(state: BWState, goals: BWGoals) -> bool:
    if goals.handempty and state.holding is not None:
        return False
    if goals.holding and state.holding != goals.holding:
        return False
    for child, parent in goals.on:
        if state.on.get(child) != parent:
            return False
    clear = _clear_blocks(state)
    for block in goals.clear:
        if block not in clear:
            return False
    for block in goals.ontable:
        if block not in state.ontable:
            return False
    return True


def _clear_blocks(state: BWState) -> Set[str]:
    supported = set(state.on.values())
    return {b for b in state.blocks if b not in supported and b != state.holding}


def _pick_up(state: BWState, plan: Optional[List[str]], block: str) -> None:
    if state.holding is not None:
        raise ValueError("pick-up requires an empty hand")
    if block not in state.ontable:
        raise ValueError(f"{block} is not on the table")
    if block not in _clear_blocks(state):
        raise ValueError(f"{block} is not clear")
    state.ontable.remove(block)
    state.holding = block
    if plan is not None:
        plan.append(f"pick-up {block}")


def _put_down(state: BWState, plan: Optional[List[str]], block: str) -> None:
    if state.holding != block:
        raise ValueError(f"arm is not holding {block}")
    state.holding = None
    state.ontable.add(block)
    if plan is not None:
        plan.append(f"put-down {block}")


def _stack(state: BWState, plan: Optional[List[str]], block: str, parent: str) -> None:
    if state.holding != block:
        raise ValueError(f"arm is not holding {block}")
    if parent not in _clear_blocks(state):
        raise ValueError(f"{parent} is not clear")
    state.holding = None
    state.on[block] = parent
    state.ontable.discard(block)
    if plan is not None:
        plan.append(f"stack {block} {parent}")


def _unstack(state: BWState, plan: Optional[List[str]], block: str, parent: str) -> None:
    if state.holding is not None:
        raise ValueError("unstack requires an empty hand")
    if state.on.get(block) != parent:
        raise ValueError(f"{block} is not on {parent}")
    if block not in _clear_blocks(state):
        raise ValueError(f"{block} is not clear")
    del state.on[block]
    state.holding = block
    if plan is not None:
        plan.append(f"unstack {block} {parent}")


def _block_key(block: str):
    try:
        return int(block.split("_")[1])
    except (IndexError, ValueError):
        return block
