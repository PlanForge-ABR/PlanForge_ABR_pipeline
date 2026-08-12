from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ArchitectSpec:
    """Executable ABR design for NaturalPlan meeting-planning instances."""

    representation: str = (
        "A meeting plan is an ordered route through friends. State is the "
        "current location, current clock minute, and the set of already met "
        "friends. Each friend has a location, availability window, and minimum "
        "meeting duration."
    )
    strategy: str = (
        "Parse structured constraints, then run deterministic dynamic search "
        "over feasible next meetings. The objective is to meet the maximum "
        "number of friends; ties prefer earlier completion and stable input "
        "order so output is reproducible."
    )


@dataclass(frozen=True)
class Friend:
    name: str
    location: str
    window_start: int
    window_end: int
    duration: int
    index: int


@dataclass(frozen=True)
class Problem:
    key: str
    start_location: str
    start_time: int
    friends: Tuple[Friend, ...]
    dist_matrix: Dict[str, Dict[str, int]]


@dataclass(frozen=True)
class Meeting:
    friend: Friend
    depart_location: str
    travel_minutes: int
    arrival_time: int
    meeting_start: int
    meeting_end: int


@dataclass(frozen=True)
class RunnerResult:
    status: str
    plan: Optional[List[str]]
    meetings: Tuple[Meeting, ...] = ()
    reason: Optional[str] = None


TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(AM|PM)\s*$", re.I)


def parse_time(text: str) -> int:
    match = TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid time: {text!r}")
    hour = int(match.group(1))
    minute = int(match.group(2))
    suffix = match.group(3).upper()
    if hour == 12:
        hour = 0
    if suffix == "PM":
        hour += 12
    return hour * 60 + minute


def format_time(minutes: int) -> str:
    minutes %= 24 * 60
    hour24, minute = divmod(minutes, 60)
    suffix = "AM" if hour24 < 12 else "PM"
    hour = hour24 % 12
    if hour == 0:
        hour = 12
    return f"{hour}:{minute:02d}{suffix}"


def parse_window(text: str) -> Tuple[int, int]:
    left, right = text.split(" to ", 1)
    return parse_time(left), parse_time(right)


def parse_problem(key: str, instance: dict) -> Problem:
    constraints = instance["constraints"]
    start_location, start_time_text = constraints[0]
    friends: List[Friend] = []
    for index, raw in enumerate(constraints[1:]):
        name, location, window, duration = raw
        window_start, window_end = parse_window(window)
        friends.append(
            Friend(
                name=name,
                location=location,
                window_start=window_start,
                window_end=window_end,
                duration=int(duration),
                index=index,
            )
        )
    return Problem(
        key=key,
        start_location=start_location,
        start_time=parse_time(start_time_text),
        friends=tuple(friends),
        dist_matrix=instance["dist_matrix"],
    )


def travel_time(problem: Problem, src: str, dst: str) -> Optional[int]:
    if src == dst:
        return 0
    value = problem.dist_matrix.get(src, {}).get(dst)
    return int(value) if value is not None else None


def solve_problem(problem: Problem) -> RunnerResult:
    meetings = search_best_schedule(problem)
    if meetings is None:
        return RunnerResult("FAILURE", None, reason="no feasible meetings")
    if not verify_meetings(problem, meetings):
        return RunnerResult("FAILURE", None, tuple(meetings), "internal verification failed")
    return RunnerResult("SUCCESS", format_plan(problem, meetings), tuple(meetings))


def search_best_schedule(problem: Problem) -> Optional[List[Meeting]]:
    friends = problem.friends
    n = len(friends)
    all_mask = (1 << n) - 1

    @lru_cache(maxsize=None)
    def dfs(mask: int, location: str, time_now: int) -> Tuple[Tuple[int, int, Tuple[int, ...]], Tuple[Meeting, ...]]:
        best_key = (0, -time_now, ())
        best_plan: Tuple[Meeting, ...] = ()

        if mask == all_mask:
            return best_key, best_plan

        remaining = [friend for friend in friends if not (mask & (1 << friend.index))]
        remaining.sort(key=lambda f: (f.window_end, f.window_start, f.index))

        for friend in remaining:
            minutes = travel_time(problem, location, friend.location)
            if minutes is None:
                continue
            arrival = time_now + minutes
            start = max(arrival, friend.window_start)
            end = start + friend.duration
            if end > friend.window_end:
                continue
            meeting = Meeting(
                friend=friend,
                depart_location=location,
                travel_minutes=minutes,
                arrival_time=arrival,
                meeting_start=start,
                meeting_end=end,
            )
            child_key, child_plan = dfs(mask | (1 << friend.index), friend.location, end)
            count = 1 + child_key[0]
            route = (friend.index,) + child_key[2]
            candidate_key = (count, -end if count == 1 else child_key[1], route)
            candidate_plan = (meeting,) + child_plan
            if better_candidate(candidate_key, candidate_plan, best_key, best_plan):
                best_key = candidate_key
                best_plan = candidate_plan

        return best_key, best_plan

    _, plan = dfs(0, problem.start_location, problem.start_time)
    return list(plan)


def better_candidate(
    candidate_key: Tuple[int, int, Tuple[int, ...]],
    candidate_plan: Tuple[Meeting, ...],
    best_key: Tuple[int, int, Tuple[int, ...]],
    best_plan: Tuple[Meeting, ...],
) -> bool:
    if candidate_key[0] != best_key[0]:
        return candidate_key[0] > best_key[0]
    if not candidate_plan:
        return False
    if not best_plan:
        return True
    candidate_finish = candidate_plan[-1].meeting_end
    best_finish = best_plan[-1].meeting_end
    if candidate_finish != best_finish:
        return candidate_finish < best_finish
    candidate_wait = total_wait(candidate_plan)
    best_wait = total_wait(best_plan)
    if candidate_wait != best_wait:
        return candidate_wait < best_wait
    return candidate_key[2] < best_key[2]


def total_wait(plan: Sequence[Meeting]) -> int:
    return sum(max(0, meeting.meeting_start - meeting.arrival_time) for meeting in plan)


def verify_meetings(problem: Problem, meetings: Sequence[Meeting]) -> bool:
    seen = set()
    location = problem.start_location
    time_now = problem.start_time
    for meeting in meetings:
        friend = meeting.friend
        if friend.index in seen:
            return False
        seen.add(friend.index)
        minutes = travel_time(problem, location, friend.location)
        if minutes is None or minutes != meeting.travel_minutes:
            return False
        if time_now + minutes != meeting.arrival_time:
            return False
        if meeting.meeting_start < meeting.arrival_time:
            return False
        if meeting.meeting_start < friend.window_start:
            return False
        if meeting.meeting_end > friend.window_end:
            return False
        if meeting.meeting_end - meeting.meeting_start < friend.duration:
            return False
        location = friend.location
        time_now = meeting.meeting_end
    return True


def optimal_meeting_count(problem: Problem) -> int:
    result = search_best_schedule(problem)
    return len(result or [])


def format_plan(problem: Problem, meetings: Sequence[Meeting]) -> List[str]:
    lines = [
        f"You start at {problem.start_location} at {format_time(problem.start_time)}."
    ]
    for meeting in meetings:
        friend = meeting.friend
        if meeting.travel_minutes:
            lines.append(
                f"You travel to {friend.location} in {meeting.travel_minutes} minutes "
                f"and arrive at {format_time(meeting.arrival_time)}."
            )
        if meeting.meeting_start > meeting.arrival_time:
            lines.append(f"You wait until {format_time(meeting.meeting_start)}.")
        lines.append(
            f"You meet {friend.name} for {friend.duration} minutes from "
            f"{format_time(meeting.meeting_start)} to {format_time(meeting.meeting_end)}."
        )
    return lines


def architect_spec() -> ArchitectSpec:
    return ArchitectSpec()


def builder_run(key: str, instance: dict) -> RunnerResult:
    problem = parse_problem(key, instance)
    return solve_problem(problem)


def runner_execute(key: str, instance: dict) -> dict:
    result = builder_run(key, instance)
    return {"status": result.status, "plan": result.plan}
