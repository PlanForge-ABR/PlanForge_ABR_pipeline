from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from typing import Callable, Hashable, Iterable, List, Optional, Tuple

from architect.state_schema import SearchNode, TripState


StateKey = Tuple[str | None, Tuple[str, ...], int]


def state_key(state: TripState) -> StateKey:
    return state.current_city, state.visited_cities, state.total_days_used


def bfs(
    initial_state: TripState,
    successor_fn: Callable[[TripState], Iterable[TripState]],
    goal_test_fn: Callable[[TripState], bool],
    max_expansions: int = 250000,
) -> Optional[SearchNode]:
    queue = deque([SearchNode(initial_state)])
    seen: set[Hashable] = {state_key(initial_state)}
    expansions = 0

    while queue and expansions < max_expansions:
        node = queue.popleft()
        if goal_test_fn(node.state):
            return node
        expansions += 1
        for child in successor_fn(node.state):
            key = state_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append(SearchNode(child, child.visited_cities))
    return None


def dfs(
    initial_state: TripState,
    successor_fn: Callable[[TripState], Iterable[TripState]],
    goal_test_fn: Callable[[TripState], bool],
    max_expansions: int = 250000,
) -> Optional[SearchNode]:
    stack: List[SearchNode] = [SearchNode(initial_state)]
    seen: set[Hashable] = set()
    expansions = 0

    while stack and expansions < max_expansions:
        node = stack.pop()
        key = state_key(node.state)
        if key in seen:
            continue
        seen.add(key)
        if goal_test_fn(node.state):
            return node
        expansions += 1
        children = list(successor_fn(node.state))
        for child in reversed(children):
            stack.append(SearchNode(child, child.visited_cities))
    return None


def astar(
    initial_state: TripState,
    successor_fn: Callable[[TripState], Iterable[TripState]],
    goal_test_fn: Callable[[TripState], bool],
    heuristic_fn: Callable[[TripState], int] | None = None,
    max_expansions: int = 250000,
) -> Optional[SearchNode]:
    heuristic = heuristic_fn or (lambda state: len(state.remaining_cities))
    serial = count()
    heap: list[tuple[int, int, int, SearchNode]] = []
    heappush(heap, (heuristic(initial_state), 0, next(serial), SearchNode(initial_state)))
    best_cost: dict[Hashable, int] = {state_key(initial_state): 0}
    expansions = 0

    while heap and expansions < max_expansions:
        _, cost, _, node = heappop(heap)
        if goal_test_fn(node.state):
            return node
        expansions += 1
        for child in successor_fn(node.state):
            child_cost = cost + 1
            key = state_key(child)
            if key in best_cost and best_cost[key] <= child_cost:
                continue
            best_cost[key] = child_cost
            score = child_cost + heuristic(child)
            heappush(heap, (score, child_cost, next(serial), SearchNode(child, child.visited_cities)))
    return None

