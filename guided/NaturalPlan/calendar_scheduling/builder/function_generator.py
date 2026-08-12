from __future__ import annotations

from typing import Iterable, Tuple

from architect.integration_contract import BuilderOutput
from architect.state_schema import CalendarAction, CalendarProblem, CalendarState


def overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def satisfies_preference(problem: CalendarProblem, day: str, start: int, end: int) -> bool:
    for preference in problem.preferences:
        if day != preference.day:
            continue
        if preference.direction == "day":
            return False
        if preference.minute is None:
            raise ValueError(f"Missing time boundary for {preference.direction} preference.")
        if preference.direction == "before" and start < preference.minute:
            return False
        if preference.direction == "after" and end > preference.minute:
            return False
        if preference.direction not in {"before", "after"}:
            raise ValueError(f"Unknown preference direction: {preference.direction}")
    return True


def slot_is_valid(problem: CalendarProblem, start: int, end: int, day: str | None = None) -> bool:
    day = day or problem.day
    if start < problem.work_start or end > problem.work_end:
        return False
    if end - start != problem.duration:
        return False
    if not satisfies_preference(problem, day, start, end):
        return False
    for person in problem.people:
        for busy_start, busy_end in problem.busy[person][day]:
            if overlaps(start, end, busy_start, busy_end):
                return False
    return True


def candidate_starts(problem: CalendarProblem) -> Tuple[tuple[str, int], ...]:
    starts = list(range(problem.work_start, problem.work_end - problem.duration + 1, 30))
    candidates: list[tuple[str, int]] = []
    after_preferences = {
        preference.day: preference
        for preference in problem.preferences
        if preference.direction == "after" and preference.minute is not None
    }
    for day in problem.days:
        day_starts = starts
        preference = after_preferences.get(day)
        if preference:
            before = [start for start in starts if start + problem.duration <= preference.minute]
            after = [start for start in starts if start + problem.duration > preference.minute]
            # Dataset references usually place "avoid after" meetings as late as possible
            # before the stated boundary, while falling back to regular order if needed.
            day_starts = list(reversed(before)) + after
        candidates.extend((day, start) for start in day_starts)
    return tuple(candidates)


def generate_builder_output(problem: CalendarProblem) -> BuilderOutput:
    initial_state = CalendarState()

    def goal_test(state: CalendarState) -> bool:
        return state.assigned and state.start is not None and state.end is not None

    def successors(state: CalendarState) -> Iterable[Tuple[CalendarAction, CalendarState]]:
        if state.assigned:
            return []
        results = []
        checked = list(state.checked_slots)
        for day, start in candidate_starts(problem):
            end = start + problem.duration
            checked.append(start)
            if not slot_is_valid(problem, start, end, day):
                continue
            action = CalendarAction(day, start, end)
            next_state = CalendarState(
                assigned=True,
                day=day,
                start=start,
                end=end,
                checked_slots=tuple(checked),
            )
            results.append((action, next_state))
        return results

    return BuilderOutput(problem, initial_state, goal_test, successors)
