"""Builder-side tests for generated meeting planners."""

from __future__ import annotations

from architect.integration_contract import PlannerBundle
from architect.state_schema import MeetingProblem


def generate_tests(problem: MeetingProblem, bundle: PlannerBundle) -> list[callable]:
    def initial_state_is_start() -> None:
        state = bundle.initial_state
        assert state.current_location == problem.start_location
        assert state.current_time == problem.start_time
        assert state.visited_people == tuple()

    def successors_do_not_repeat_people() -> None:
        for state in bundle.successor_fn(bundle.initial_state):
            assert len(state.visited_people) == len(set(state.visited_people))

    def successors_respect_windows_and_duration() -> None:
        for state in bundle.successor_fn(bundle.initial_state):
            action = state.actions[-1]
            friend = problem.friends[action.friend_index]
            assert action.meeting_start >= friend.window_start
            assert action.meeting_end <= friend.window_end
            assert action.meeting_end - action.meeting_start == friend.min_duration

    return [initial_state_is_start, successors_do_not_repeat_people, successors_respect_windows_and_duration]

