def combinations_func(data):
    from math import gcd
    from functools import reduce

    def parse_time_to_minutes(t):
        if isinstance(t, int):
            return t
        if isinstance(t, str):
            parts = t.strip().split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        raise ValueError(f"Unsupported time format: {t}")

    def parse_duration_to_minutes(d):
        if isinstance(d, int):
            return d
        if isinstance(d, str):
            parts = d.strip().split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        raise ValueError(f"Unsupported duration format: {d}")

    def minutes_to_time(m):
        return f"{m // 60}:{m % 60:02d}"

    participants = data.get("participants", [])
    day = data.get("day")
    work_hours = data.get("work_hours", {})
    existing_schedules = data.get("existing_schedules", {})
    meeting_duration = parse_duration_to_minutes(data.get("meeting_duration"))

    work_start = parse_time_to_minutes(work_hours.get("start"))
    work_end = parse_time_to_minutes(work_hours.get("end"))

    busy_intervals = []
    granular_points = [meeting_duration, work_start, work_end]

    for person in participants:
        for interval in existing_schedules.get(person, []):
            s = parse_time_to_minutes(interval.get("start"))
            e = parse_time_to_minutes(interval.get("end"))
            busy_intervals.append((s, e))
            granular_points.extend([s, e])

    positive_points = [abs(x) for x in granular_points if isinstance(x, int) and x != 0]
    step = reduce(gcd, positive_points) if positive_points else 1
    if step <= 0:
        step = 1

    candidates = []
    latest_start = work_end - meeting_duration

    for start in range(work_start, latest_start + 1, step):
        end = start + meeting_duration

        if end > work_end:
            continue

        conflict = False
        for b_start, b_end in busy_intervals:
            if start < b_end and end > b_start:
                conflict = True
                break

        if not conflict:
            candidates.append([
                {
                    "day": day,
                    "start": minutes_to_time(start),
                    "end": minutes_to_time(end),
                }
            ])

    return candidates