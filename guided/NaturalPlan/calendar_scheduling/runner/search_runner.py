from __future__ import annotations

from architect.integration_contract import BuilderOutput
from architect.search_library import astar, bfs, dfs
from architect.state_schema import CalendarAction
from runner.executor import load_builder_output


SEARCH_ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "astar": astar,
}


def run_search(output: BuilderOutput, algorithm: str = "bfs") -> CalendarAction | None:
    loaded = load_builder_output(output)
    search = SEARCH_ALGORITHMS[algorithm]
    plan = search(loaded.initial_state, loaded.successor_fn, loaded.goal_test)
    if not plan:
        return None
    return plan[-1]

