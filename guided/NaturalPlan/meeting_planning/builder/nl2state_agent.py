"""Convert meeting-planning examples into validated structured data."""

from __future__ import annotations

import re
from typing import Any

from architect.state_schema import FriendConstraint, MeetingProblem

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(AM|PM)$")


def parse_time(text: str) -> int:
    match = TIME_RE.match(text.strip())
    if not match:
        raise ValueError(f"Invalid time: {text!r}")
    hour = int(match.group(1))
    minute = int(match.group(2))
    suffix = match.group(3)
    if hour == 12:
        hour = 0
    if suffix == "PM":
        hour += 12
    return hour * 60 + minute


def format_time(minutes: int) -> str:
    minutes %= 24 * 60
    hour_24, minute = divmod(minutes, 60)
    suffix = "AM" if hour_24 < 12 else "PM"
    hour = hour_24 % 12
    if hour == 0:
        hour = 12
    return f"{hour}:{minute:02d}{suffix}"


def parse_window(text: str) -> tuple[int, int]:
    start, end = text.split(" to ", 1)
    return parse_time(start), parse_time(end)


def nl2state_agent(example_id: str, example: dict[str, Any]) -> MeetingProblem:
    constraints = example.get("constraints")
    if not isinstance(constraints, list) or len(constraints) < 1:
        raise ValueError(f"{example_id}: constraints must include a start constraint")

    start_constraint = constraints[0]
    if len(start_constraint) != 2:
        raise ValueError(f"{example_id}: invalid start constraint")
    start_location, start_time_text = start_constraint
    start_time = parse_time(start_time_text)

    friends = []
    for index, raw in enumerate(constraints[1:]):
        if len(raw) != 4:
            raise ValueError(f"{example_id}: invalid friend constraint at index {index}")
        name, location, window_text, min_duration = raw
        window_start, window_end = parse_window(window_text)
        if int(min_duration) <= 0:
            raise ValueError(f"{example_id}: duration must be positive")
        friends.append(
            FriendConstraint(
                index=index,
                name=str(name),
                location=str(location),
                window_start=window_start,
                window_end=window_end,
                min_duration=int(min_duration),
            )
        )

    dist_matrix = example.get("dist_matrix")
    if not isinstance(dist_matrix, dict):
        raise ValueError(f"{example_id}: dist_matrix must be a dict")
    for origin, destinations in dist_matrix.items():
        if not isinstance(destinations, dict):
            raise ValueError(f"{example_id}: bad distance row for {origin}")
        for destination, minutes in destinations.items():
            if not isinstance(minutes, int) or minutes < 0:
                raise ValueError(f"{example_id}: bad distance {origin}->{destination}")

    return MeetingProblem(
        example_id=example_id,
        start_location=str(start_location),
        start_time=start_time,
        friends=tuple(friends),
        dist_matrix=dist_matrix,
    )

