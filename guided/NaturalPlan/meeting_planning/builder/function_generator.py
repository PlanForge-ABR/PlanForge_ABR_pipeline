"""Generate meeting-specific planning functions from structured data."""

from __future__ import annotations

from architect.integration_contract import PlannerBundle
from architect.state_schema import MeetingAction, MeetingProblem, MeetingState
from builder.nl2state_agent import format_time


def generate_functions(problem: MeetingProblem) -> PlannerBundle:
    initial_state = MeetingState(
        current_location=problem.start_location,
        current_time=problem.start_time,
        visited_people=tuple(),
        actions=tuple(),
    )

    def goal_test(state: MeetingState) -> bool:
        return len(state.visited_people) == len(problem.friends)

    def successor_fn(state: MeetingState) -> list[MeetingState]:
        successors: list[MeetingState] = []
        visited = set(state.visited_people)
        for friend in problem.friends:
            if friend.index in visited:
                continue
            travel_minutes = problem.dist_matrix.get(state.current_location, {}).get(friend.location)
            if travel_minutes is None:
                continue
            arrival_time = state.current_time + travel_minutes
            meeting_start = max(arrival_time, friend.window_start)
            meeting_end = meeting_start + friend.min_duration
            if meeting_end > friend.window_end:
                continue
            action = MeetingAction(
                friend_index=friend.index,
                friend_name=friend.name,
                origin=state.current_location,
                destination=friend.location,
                depart_time=state.current_time,
                travel_minutes=travel_minutes,
                arrival_time=arrival_time,
                wait_until=meeting_start if meeting_start > arrival_time else None,
                meeting_start=meeting_start,
                meeting_end=meeting_end,
                duration=friend.min_duration,
            )
            successors.append(
                MeetingState(
                    current_location=friend.location,
                    current_time=meeting_end,
                    visited_people=state.visited_people + (friend.index,),
                    actions=state.actions + (action,),
                )
            )
        return successors

    def score_fn(state: MeetingState) -> tuple:
        # Maximize meetings, then preserve the earliest deterministic sequence from constraints.
        return (len(state.visited_people), -sum((i + 1) * index for i, index in enumerate(state.visited_people)))

    def formatter(state: MeetingState) -> list[str]:
        lines = [f"You start at {problem.start_location} at {format_time(problem.start_time)}."]
        for action in state.actions:
            lines.append(
                f"You travel to {action.destination} in {action.travel_minutes} minutes "
                f"and arrive at {format_time(action.arrival_time)}."
            )
            if action.wait_until is not None:
                lines.append(f"You wait until {format_time(action.wait_until)}.")
            lines.append(
                f"You meet {action.friend_name} for {action.duration} minutes from "
                f"{format_time(action.meeting_start)} to {format_time(action.meeting_end)}."
            )
        return lines

    return PlannerBundle(
        initial_state=initial_state,
        goal_test=goal_test,
        successor_fn=successor_fn,
        score_fn=score_fn,
        formatter=formatter,
    )

