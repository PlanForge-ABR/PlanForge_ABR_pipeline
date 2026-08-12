from __future__ import annotations

from architect.state_schema import CalendarProblem
from builder.function_generator import candidate_starts, slot_is_valid


def generate_and_run_tests(problem: CalendarProblem) -> list[str]:
    errors: list[str] = []
    starts = candidate_starts(problem)
    if not starts:
        errors.append("No candidate starts generated.")
    for _day, start in starts:
        end = start + problem.duration
        if end - start != problem.duration:
            errors.append("Candidate duration mismatch.")
        if start % 30 != 0 or end % 30 != 0:
            errors.append("Candidate is not on 30-minute grid.")
    for person, day_map in problem.busy.items():
        for day, intervals in day_map.items():
            for busy_start, busy_end in intervals:
                midpoint = busy_start
                if problem.work_start <= midpoint <= problem.work_end - problem.duration:
                    if slot_is_valid(problem, midpoint, midpoint + problem.duration, day):
                        errors.append(f"Busy overlap accepted for {person} on {day}.")
    return errors
