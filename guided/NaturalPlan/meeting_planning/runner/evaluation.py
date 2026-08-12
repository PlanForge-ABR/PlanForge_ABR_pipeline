"""Evaluation metrics for meeting-planning predictions."""

from __future__ import annotations

import re

from architect.state_schema import MeetingProblem
from builder.nl2state_agent import parse_time

TRAVEL_RE = re.compile(r"^You travel to (.+) in (\d+) minutes and arrive at (.+)\.$")
WAIT_RE = re.compile(r"^You wait until (.+)\.$")
MEET_RE = re.compile(r"^You meet (.+) for (\d+) minutes from (.+) to (.+)\.$")


def exact_match(predicted_plan: list[str], golden_plan: list[str]) -> bool:
    return predicted_plan == golden_plan


def constraint_satisfied(problem: MeetingProblem, predicted_plan: list[str]) -> bool:
    if not predicted_plan:
        return False
    expected_start = f"You start at {problem.start_location} at "
    if predicted_plan[0] != f"{expected_start}{_format_problem_start(problem)}.":
        return False

    current_location = problem.start_location
    current_time = problem.start_time
    met: set[str] = set()
    i = 1
    while i < len(predicted_plan):
        travel_match = TRAVEL_RE.match(predicted_plan[i])
        if not travel_match:
            return False
        destination = travel_match.group(1)
        travel_minutes = int(travel_match.group(2))
        arrival_time = parse_time(travel_match.group(3))
        expected_travel = problem.dist_matrix.get(current_location, {}).get(destination)
        if expected_travel != travel_minutes:
            return False
        if current_time + travel_minutes != arrival_time:
            return False
        current_location = destination
        current_time = arrival_time
        i += 1

        if i < len(predicted_plan):
            wait_match = WAIT_RE.match(predicted_plan[i])
            if wait_match:
                wait_until = parse_time(wait_match.group(1))
                if wait_until < current_time:
                    return False
                current_time = wait_until
                i += 1

        if i >= len(predicted_plan):
            return False
        meet_match = MEET_RE.match(predicted_plan[i])
        if not meet_match:
            return False
        name = meet_match.group(1)
        duration = int(meet_match.group(2))
        start = parse_time(meet_match.group(3))
        end = parse_time(meet_match.group(4))
        matches = [friend for friend in problem.friends if friend.name == name]
        if not matches:
            return False
        friend = matches[0]
        if friend.location != current_location:
            return False
        if duration < friend.min_duration:
            return False
        if end - start != duration:
            return False
        if start != current_time:
            return False
        if start < friend.window_start or end > friend.window_end:
            return False
        if name in met:
            return False
        met.add(name)
        current_time = end
        i += 1
    return len(met) == _max_feasible_meetings(problem)


def _format_problem_start(problem: MeetingProblem) -> str:
    from builder.nl2state_agent import format_time

    return format_time(problem.start_time)


def _max_feasible_meetings(problem: MeetingProblem) -> int:
    best = 0

    def visit(location: str, time: int, visited: frozenset[int]) -> None:
        nonlocal best
        best = max(best, len(visited))
        for friend in problem.friends:
            if friend.index in visited:
                continue
            travel = problem.dist_matrix.get(location, {}).get(friend.location)
            if travel is None:
                continue
            arrival = time + travel
            start = max(arrival, friend.window_start)
            end = start + friend.min_duration
            if end <= friend.window_end:
                visit(friend.location, end, visited | {friend.index})

    visit(problem.start_location, problem.start_time, frozenset())
    return best


def summarize(results: list[dict]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {"total": 0, "SR": 0.0, "EM": 0.0}
    sr = sum(1 for row in results if row["success"]) / total
    em = sum(1 for row in results if row["exact_match"]) / total
    return {"total": total, "SR": sr, "EM": em}
