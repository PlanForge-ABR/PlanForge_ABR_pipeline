"""Deterministic search algorithms used by the runner."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from heapq import heappop, heappush
from typing import TypeVar

T = TypeVar("T")


def bfs(initial_state: T, successor_fn: Callable[[T], Iterable[T]], goal_test_fn: Callable[[T], bool]) -> T | None:
    frontier = deque([initial_state])
    seen = {initial_state}
    while frontier:
        state = frontier.popleft()
        if goal_test_fn(state):
            return state
        for successor in successor_fn(state):
            if successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return None


def dfs(initial_state: T, successor_fn: Callable[[T], Iterable[T]], goal_test_fn: Callable[[T], bool]) -> T | None:
    frontier = [initial_state]
    seen = {initial_state}
    while frontier:
        state = frontier.pop()
        if goal_test_fn(state):
            return state
        for successor in successor_fn(state):
            if successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return None


def astar(
    initial_state: T,
    successor_fn: Callable[[T], Iterable[T]],
    goal_test_fn: Callable[[T], bool],
    heuristic_fn: Callable[[T], int],
) -> T | None:
    counter = 0
    frontier: list[tuple[int, int, T]] = []
    heappush(frontier, (heuristic_fn(initial_state), counter, initial_state))
    best_cost = {initial_state: 0}
    while frontier:
        _, _, state = heappop(frontier)
        if goal_test_fn(state):
            return state
        cost = best_cost[state] + 1
        for successor in successor_fn(state):
            if successor not in best_cost or cost < best_cost[successor]:
                best_cost[successor] = cost
                counter += 1
                heappush(frontier, (cost + heuristic_fn(successor), counter, successor))
    return None


def exhaustive_best(
    initial_state: T,
    successor_fn: Callable[[T], Iterable[T]],
    score_fn: Callable[[T], tuple],
) -> T:
    best = initial_state
    frontier = [initial_state]
    seen = {initial_state}
    while frontier:
        state = frontier.pop()
        if score_fn(state) > score_fn(best):
            best = state
        successors = list(successor_fn(state))
        for successor in reversed(successors):
            if successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return best

