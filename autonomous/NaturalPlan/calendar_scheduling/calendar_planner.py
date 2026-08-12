from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
DAY_RE = "|".join(DAY_ORDER)
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


@dataclass(frozen=True)
class ArchitectSpec:
    """Executable ABR design for NaturalPlan calendar-scheduling instances."""

    representation: str = (
        "A calendar problem is represented as candidate meeting slots over an "
        "ordered set of work days. Each participant has busy intervals and "
        "optional day/time exclusions parsed from the prompt."
    )
    strategy: str = (
        "The builder parses the natural-language task into structured working "
        "hours, duration, busy intervals, and preference constraints. The "
        "runner scans deterministic 30-minute grid starts in chronological "
        "order and returns the first slot that satisfies every constraint."
    )


@dataclass(frozen=True)
class Problem:
    key: str
    participants: Tuple[str, ...]
    days: Tuple[str, ...]
    work_start: int
    work_end: int
    duration: int
    busy: Dict[str, Dict[str, Tuple[Tuple[int, int], ...]]]
    not_before: Dict[str, Dict[str, int]] = field(default_factory=dict)
    not_after: Dict[str, Dict[str, int]] = field(default_factory=dict)
    blocked_days: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    prefer_earliest: bool = True


@dataclass(frozen=True)
class RunnerResult:
    status: str
    plan: Optional[str]
    day: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    reason: Optional[str] = None


def parse_clock(text: str) -> int:
    match = TIME_RE.search(text.strip())
    if not match:
        raise ValueError(f"invalid time: {text!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def format_clock(minutes: int) -> str:
    hour, minute = divmod(minutes, 60)
    return f"{hour}:{minute:02d}"


def parse_duration(text: str) -> int:
    text = text.lower().strip()
    if text == "half an hour":
        return 30
    if text == "one hour":
        return 60
    match = re.search(r"(\d+)\s+minutes?", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"unsupported duration: {text!r}")


def split_people(text: str) -> Tuple[str, ...]:
    parts = [part.strip() for part in re.split(r",|\band\b", text) if part.strip()]
    return tuple(parts)


def parse_days(text: str) -> Tuple[str, ...]:
    days = [day for day in DAY_ORDER if re.search(rf"\b{day}\b", text)]
    if not days:
        raise ValueError(f"no days found in {text!r}")
    return tuple(days)


def task_text(instance: dict) -> str:
    prompt = instance["prompt_0shot"]
    return prompt.split("TASK:")[-1]


def parse_problem(key: str, instance: dict) -> Problem:
    task = task_text(instance)
    header_match = re.search(
        r"You need to schedule a meeting for (?P<people>.+?) for "
        r"(?P<duration>half an hour|one hour|\d+ minutes?) between the "
        r"work hours of (?P<start>\d{1,2}:\d{2}) to (?P<end>\d{1,2}:\d{2}) "
        r"on (?P<days>.+?)\.",
        task,
        re.S,
    )
    if not header_match:
        raise ValueError(f"could not parse task header for {key}")

    participants = split_people(header_match.group("people"))
    days = parse_days(header_match.group("days"))
    busy = {person: {day: tuple() for day in days} for person in participants}

    schedule_block = task
    find_index = schedule_block.find("Find a time")
    if find_index >= 0:
        schedule_block = schedule_block[:find_index]

    for person in participants:
        line = find_person_line(schedule_block, person)
        if not line or is_open_calendar_line(line):
            continue
        person_busy = parse_busy_line(line, days)
        for day, intervals in person_busy.items():
            busy[person][day] = tuple(sorted(merge_intervals(intervals)))

    not_before: Dict[str, Dict[str, int]] = {person: {} for person in participants}
    not_after: Dict[str, Dict[str, int]] = {person: {} for person in participants}
    blocked_days: Dict[str, Tuple[str, ...]] = {person: tuple() for person in participants}
    parse_preferences(task, participants, days, not_before, not_after, blocked_days)

    return Problem(
        key=key,
        participants=participants,
        days=days,
        work_start=parse_clock(header_match.group("start")),
        work_end=parse_clock(header_match.group("end")),
        duration=parse_duration(header_match.group("duration")),
        busy=busy,
        not_before=not_before,
        not_after=not_after,
        blocked_days=blocked_days,
        prefer_earliest=("earlist availability" in task or "earliest availability" in task),
    )


def find_person_line(block: str, person: str) -> Optional[str]:
    escaped = re.escape(person)
    match = re.search(rf"(?m)^\s*{escaped}\b.*?(?:;|\.)(?:\s*$)?", block)
    if match:
        return match.group(0).strip()
    match = re.search(rf"{escaped}(?:has|is|\'s).*?(?:;|\.)", block)
    return match.group(0).strip() if match else None


def is_open_calendar_line(line: str) -> bool:
    lowered = line.lower()
    return (
        "wide open the entire day" in lowered
        or "free the entire day" in lowered
        or "no meetings the whole day" in lowered
    )


def parse_busy_line(line: str, days: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    parsed = {day: [] for day in days}
    pattern = re.compile(
        rf"(?:on\s+)?(?P<day>{DAY_RE})\s+during\s+"
        rf"(?P<intervals>.*?)(?=,\s*(?:{DAY_RE})\s+during|;|\.|$)",
        re.S,
    )
    for match in pattern.finditer(line):
        day = match.group("day")
        if day not in parsed:
            continue
        parsed[day].extend(parse_intervals(match.group("intervals")))
    return parsed


def parse_intervals(text: str) -> List[Tuple[int, int]]:
    intervals = []
    for start, end in re.findall(r"(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})", text):
        intervals.append((parse_clock(start), parse_clock(end)))
    return intervals


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def parse_preferences(
    task: str,
    participants: Sequence[str],
    days: Sequence[str],
    not_before: Dict[str, Dict[str, int]],
    not_after: Dict[str, Dict[str, int]],
    blocked_days: Dict[str, Tuple[str, ...]],
) -> None:
    verb = (
        r"(?:would rather not meet|would like to avoid more meetings|"
        r"do not want to meet|can not meet)"
    )
    for person in participants:
        person_pattern = re.escape(person)
        other_starts = [
            rf"{re.escape(other)}\s+{verb}\s+on\s+"
            for other in participants
            if other != person
        ]
        stop = "|".join(other_starts + [r"You would", r"Find a time", r"SOLUTION:"])
        pattern = re.compile(rf"{person_pattern}\s+{verb}\s+on\s+(?P<body>.*?)(?={stop}|$)", re.S)
        for match in pattern.finditer(task):
            for day, direction, time_text in re.findall(
                rf"\b({DAY_RE})\b(?:\s+(before|after)\s+(\d{{1,2}}:\d{{2}}))?",
                match.group("body"),
            ):
                if day not in days:
                    continue
                if not direction:
                    blocked_days[person] = tuple(
                        sorted(set(blocked_days[person]) | {day}, key=DAY_ORDER.index)
                    )
                    continue
                boundary = parse_clock(time_text)
                if direction == "before":
                    not_before[person][day] = max(not_before[person].get(day, 0), boundary)
                else:
                    not_after[person][day] = min(not_after[person].get(day, 24 * 60), boundary)


def solve_problem(problem: Problem) -> RunnerResult:
    slot = find_first_slot(problem)
    if slot is None:
        return RunnerResult("FAILURE", None, reason="no valid slot found")
    day, start = slot
    end = start + problem.duration
    plan = f"Here is the proposed time: {day}, {format_clock(start)} - {format_clock(end)} "
    result = RunnerResult("SUCCESS", plan, day=day, start=start, end=end)
    if not verify_result(problem, result):
        return RunnerResult("FAILURE", None, day=day, start=start, end=end, reason="internal verification failed")
    return result


def find_first_slot(problem: Problem) -> Optional[Tuple[str, int]]:
    step = 30
    after_constraint = any(person_limits for person_limits in problem.not_after.values())
    before_constraint = any(person_limits for person_limits in problem.not_before.values())

    for day in problem.days:
        starts = [
            start
            for start in range(problem.work_start, problem.work_end - problem.duration + 1, step)
            if is_slot_valid(problem, day, start, start + problem.duration)
        ]
        if not starts:
            continue
        if after_constraint and not before_constraint and not problem.prefer_earliest:
            return day, first_start_of_last_free_block(starts, step)
        return day, starts[0]
    return None


def first_start_of_last_free_block(starts: Sequence[int], step: int) -> int:
    block_start = starts[0]
    previous = starts[0]
    for start in starts[1:]:
        if start != previous + step:
            block_start = start
        previous = start
    return block_start


def is_slot_valid(problem: Problem, day: str, start: int, end: int) -> bool:
    if start < problem.work_start or end > problem.work_end:
        return False
    for person in problem.participants:
        if day in problem.blocked_days.get(person, ()):
            return False
        if start < problem.not_before.get(person, {}).get(day, problem.work_start):
            return False
        if end > problem.not_after.get(person, {}).get(day, problem.work_end):
            return False
        for busy_start, busy_end in problem.busy.get(person, {}).get(day, ()):
            if start < busy_end and end > busy_start:
                return False
    return True


def verify_result(problem: Problem, result: RunnerResult) -> bool:
    if result.status != "SUCCESS" or result.day is None or result.start is None or result.end is None:
        return False
    if result.end - result.start != problem.duration:
        return False
    return is_slot_valid(problem, result.day, result.start, result.end)


def parse_plan_slot(plan: object) -> Optional[Tuple[str, int, int]]:
    if plan is None:
        return None
    text = plan if isinstance(plan, str) else "\n".join(map(str, plan))
    match = re.search(
        rf"(?P<day>{DAY_RE}),\s*(?P<start>\d{{1,2}}:\d{{2}})\s*-\s*(?P<end>\d{{1,2}}:\d{{2}})",
        text,
    )
    if not match:
        return None
    return match.group("day"), parse_clock(match.group("start")), parse_clock(match.group("end"))


def verify_plan_text(problem: Problem, plan: object) -> bool:
    slot = parse_plan_slot(plan)
    if slot is None:
        return False
    day, start, end = slot
    result = RunnerResult("SUCCESS", str(plan), day=day, start=start, end=end)
    return verify_result(problem, result)


def normalize_plan(plan: object) -> str:
    if plan is None:
        return ""
    text = str(plan).strip()
    if text.startswith("SOLUTION:"):
        text = text[len("SOLUTION:") :].strip()
    return re.sub(r"\s+", " ", text)


def architect_spec() -> ArchitectSpec:
    return ArchitectSpec()


def builder_run(key: str, instance: dict) -> RunnerResult:
    problem = parse_problem(key, instance)
    return solve_problem(problem)


def runner_execute(key: str, instance: dict) -> dict:
    result = builder_run(key, instance)
    return {"status": result.status, "plan": result.plan}
