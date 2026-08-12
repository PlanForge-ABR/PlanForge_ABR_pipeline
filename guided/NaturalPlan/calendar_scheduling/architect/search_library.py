from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
SuccessorFn = Callable[[StateT], Iterable[Tuple[ActionT, StateT]]]
GoalFn = Callable[[StateT], bool]


def bfs(
    initial_state: StateT,
    successor_fn: SuccessorFn[StateT, ActionT],
    goal_test_fn: GoalFn[StateT],
) -> Optional[List[ActionT]]:
    frontier = deque([(initial_state, [])])
    seen = {initial_state}
    while frontier:
        state, plan = frontier.popleft()
        if goal_test_fn(state):
            return plan
        for action, next_state in successor_fn(state):
            if next_state in seen:
                continue
            seen.add(next_state)
            frontier.append((next_state, plan + [action]))
    return None


def dfs(
    initial_state: StateT,
    successor_fn: SuccessorFn[StateT, ActionT],
    goal_test_fn: GoalFn[StateT],
) -> Optional[List[ActionT]]:
    frontier = [(initial_state, [])]
    seen = {initial_state}
    while frontier:
        state, plan = frontier.pop()
        if goal_test_fn(state):
            return plan
        for action, next_state in successor_fn(state):
            if next_state in seen:
                continue
            seen.add(next_state)
            frontier.append((next_state, plan + [action]))
    return None


def astar(
    initial_state: StateT,
    successor_fn: SuccessorFn[StateT, ActionT],
    goal_test_fn: GoalFn[StateT],
    heuristic_fn: Callable[[StateT], float] | None = None,
) -> Optional[List[ActionT]]:
    heuristic = heuristic_fn or (lambda _state: 0.0)
    tie = count()
    frontier = [(heuristic(initial_state), 0, next(tie), initial_state, [])]
    best_cost = {initial_state: 0}
    while frontier:
        _priority, cost, _tie, state, plan = heappop(frontier)
        if goal_test_fn(state):
            return plan
        if cost != best_cost.get(state):
            continue
        for action, next_state in successor_fn(state):
            next_cost = cost + 1
            if next_cost >= best_cost.get(next_state, 10**9):
                continue
            best_cost[next_state] = next_cost
            heappush(
                frontier,
                (next_cost + heuristic(next_state), next_cost, next(tie), next_state, plan + [action]),
            )
    return None

