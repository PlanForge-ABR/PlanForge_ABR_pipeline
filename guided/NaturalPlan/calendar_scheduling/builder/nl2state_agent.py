from __future__ import annotations

import re
from typing import Dict, List, Tuple

from architect.state_schema import CalendarProblem, Interval, Preference


TIME_RE = re.compile(r"(\d{1,2}:\d{2})")
BUSY_LINE_RE = re.compile(
    r"^\s*(?P<person>[A-Z][A-Za-z]+)\s*(?:'s calendar is wide open the entire day|"
    r"is free the entire day|has no meetings the whole day|"
    r"(?:has meetings|has blocked their calendar|is busy) on (?P<day>[A-Za-z]+) during (?P<times>.*));?\s*$"
)
PREF_RE = re.compile(
    r"(?P<person>[A-Z][A-Za-z]+)\s+"
    r"(?:(?P<hard>can not)|(?:do not want|would rather not|would like to avoid more meetings))"
    r"\s+(?:(?:to\s+)?meet\s+)?on\s+(?P<day>[A-Za-z]+)\s+(?P<direction>before|after)\s+(?P<time>\d{1,2}:\d{2})",
    re.IGNORECASE,
)
DAY_PREF_RE = re.compile(
    r"(?P<person>[A-Z][A-Za-z]+)\s+"
    r"(?:(?P<hard>can not)|(?:do not want|would rather not|would like to avoid more meetings))"
    r"\s+(?:(?:to\s+)?meet\s+)?on\s+(?P<day>[A-Za-z]+)(?!\s+(?:before|after))",
    re.IGNORECASE,
)


def time_to_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def minutes_to_time(value: int) -> str:
    return f"{value // 60}:{value % 60:02d}"


def duration_to_minutes(duration_text: str, dataset_duration: str | None = None) -> int:
    text = duration_text.lower()
    if "half an hour" in text:
        return 30
    match = re.search(r"for\s+(.+?)\s+between", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        number = re.search(r"(\d+(?:\.\d+)?)", raw)
        if number:
            return int(float(number.group(1)) * 60)
        if "one hour" in raw:
            return 60
    if dataset_duration is not None:
        return int(float(dataset_duration) * 60)
    raise ValueError("Could not parse meeting duration.")


def _task_text(prompt: str) -> str:
    return prompt.split("TASK:")[-1].split("SOLUTION:")[0].strip()


def _parse_people(task: str) -> Tuple[str, ...]:
    match = re.search(r"meeting for (.+?) for ", task, re.DOTALL)
    if not match:
        raise ValueError("Could not parse participant list.")
    people_text = match.group(1).replace("\n", " ")
    people_text = people_text.replace(" and ", ", ")
    return tuple(part.strip() for part in people_text.split(",") if part.strip())


def _parse_work_hours(task: str) -> Tuple[int, int, Tuple[str, ...]]:
    match = re.search(
        r"between the work hours of (?P<start>\d{1,2}:\d{2}) to (?P<end>\d{1,2}:\d{2}) on (?:either )?(?P<days>[A-Za-z,\s]+?)(?=\.)",
        task,
    )
    if not match:
        raise ValueError("Could not parse work hours.")
    days = tuple(part.strip() for part in re.split(r"\s*,\s*|\s+or\s+", match.group("days")) if part.strip())
    return time_to_minutes(match.group("start")), time_to_minutes(match.group("end")), days


def _parse_busy(task: str, people: Tuple[str, ...], days: Tuple[str, ...]) -> Dict[str, Dict[str, Tuple[Interval, ...]]]:
    busy: Dict[str, Dict[str, List[Interval]]] = {
        person: {day: [] for day in days} for person in people
    }
    for raw_line in task.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Some dataset lines omit a space, e.g. "Jerryhas no meetings".
        for person in people:
            line = re.sub(rf"\b{re.escape(person)}has\b", f"{person} has", line)
        match = BUSY_LINE_RE.match(line)
        if not match:
            continue
        person = match.group("person")
        if person not in busy:
            continue
        times = match.groupdict().get("times")
        if not times:
            busy[person] = {day: [] for day in days}
            continue
        for index, day in enumerate(days):
            day_match = re.search(
                rf"{day}\s+during\s+(.*?)(?=(?:,\s*(?:{'|'.join(days)})\s+during)|;|$)",
                line,
            )
            if not day_match:
                continue
            intervals = []
            for start, end in re.findall(r"(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})", day_match.group(1)):
                intervals.append((time_to_minutes(start), time_to_minutes(end)))
            busy[person][day] = intervals
    return {
        person: {day: tuple(sorted(intervals)) for day, intervals in day_map.items()}
        for person, day_map in busy.items()
    }


def _parse_preferences(task: str) -> Tuple[Preference, ...]:
    preferences: list[Preference] = []
    spans: list[tuple[int, int]] = []
    for match in PREF_RE.finditer(task):
        preferences.append(
            Preference(
                person=match.group("person"),
                day=match.group("day"),
                direction=match.group("direction").lower(),
                minute=time_to_minutes(match.group("time")),
                hard=bool(match.group("hard")),
            )
        )
        spans.append(match.span())
    for match in DAY_PREF_RE.finditer(task):
        if any(start <= match.start() < end for start, end in spans):
            continue
        preferences.append(
            Preference(
                person=match.group("person"),
                day=match.group("day"),
                direction="day",
                minute=None,
                hard=bool(match.group("hard")),
            )
        )
    return tuple(preferences)


def parse_calendar_problem(example: dict) -> CalendarProblem:
    prompt = example["prompt_0shot"]
    task = _task_text(prompt)
    people = _parse_people(task)
    work_start, work_end, days = _parse_work_hours(task)
    duration = duration_to_minutes(task, example.get("duration"))
    problem = CalendarProblem(
        people=people,
        day=days[0],
        days=days,
        work_start=work_start,
        work_end=work_end,
        duration=duration,
        busy=_parse_busy(task, people, days),
        preferences=_parse_preferences(task),
    )
    validate_problem(problem)
    return problem


def validate_problem(problem: CalendarProblem) -> None:
    if problem.duration % 30 != 0:
        raise ValueError("Calendar scheduling expects 30-minute granularity.")
    for person, day_map in problem.busy.items():
        for day, intervals in day_map.items():
            for start, end in intervals:
                if start >= end:
                    raise ValueError(f"Invalid busy interval for {person} on {day}: {start}-{end}.")
                if start < problem.work_start or end > problem.work_end:
                    raise ValueError(f"Busy interval outside work hours for {person} on {day}.")
