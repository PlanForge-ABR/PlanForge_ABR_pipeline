from __future__ import annotations

from architect.integration_contract import BuilderOutput
from architect.search_library import astar, bfs, dfs
from architect.state_schema import SearchNode


SEARCH_ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "astar": astar,
}


def run_search(output: BuilderOutput, algorithm: str = "bfs", max_expansions: int = 250000) -> SearchNode | None:
    if algorithm not in SEARCH_ALGORITHMS:
        raise ValueError(f"Unknown search algorithm: {algorithm}")
    search_fn = SEARCH_ALGORITHMS[algorithm]
    return search_fn(
        output.initial_state,
        output.successor_fn,
        output.goal_test,
        max_expansions=max_expansions,
    )


def format_plan(output: BuilderOutput, node: SearchNode | None) -> str:
    if node is None:
        return "No plan found."

    problem = output.problem
    lines = [
        f"Here is the trip plan for visiting the {problem.num_cities} European cities for {problem.total_days} days:",
        "",
    ]
    segments = node.state.segments
    for index, segment in enumerate(segments):
        if index == 0:
            lines.append(
                f"**Day {segment.start_day}-{segment.end_day}:** "
                f"Arriving in {segment.city} and visit {segment.city} for {segment.duration} days."
            )
        else:
            previous = segments[index - 1]
            lines.append(f"**Day {segment.start_day}:** Fly from {previous.city} to {segment.city}.")
            lines.append(
                f"**Day {segment.start_day}-{segment.end_day}:** "
                f"Visit {segment.city} for {segment.duration} days."
            )
    return "\n".join(lines)

