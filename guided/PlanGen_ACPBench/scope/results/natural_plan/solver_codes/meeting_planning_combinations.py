def combinations_func(data):
    from copy import deepcopy

    def parse_time(t):
        s = t.strip().upper()
        ampm = s[-2:]
        hm = s[:-2]
        hour_str, minute_str = hm.split(":")
        hour = int(hour_str)
        minute = int(minute_str)
        if ampm == "AM":
            if hour == 12:
                hour = 0
        elif ampm == "PM":
            if hour != 12:
                hour += 12
        return hour * 60 + minute

    def format_time(minutes):
        minutes = minutes % (24 * 60)
        hour24 = minutes // 60
        minute = minutes % 60
        ampm = "AM" if hour24 < 12 else "PM"
        hour12 = hour24 % 12
        if hour12 == 0:
            hour12 = 12
        return f"{hour12}:{minute:02d}{ampm}"

    travel_lookup = {}
    for item in data.get("travel_times_minutes", []):
        travel_lookup[(item["from"], item["to"])] = item["minutes"]

    availability_by_person = {}
    for item in data.get("availability", []):
        availability_by_person.setdefault(item["person"], []).append({
            "location": item["location"],
            "start": parse_time(item["start_time"]),
            "end": parse_time(item["end_time"]),
        })

    min_meeting_by_person = {
        item["person"]: item["minutes"]
        for item in data.get("minimum_meeting_times_minutes", [])
    }

    people = list(data.get("people", []))
    arrival = data.get("arrival", {})
    start_location = arrival.get("location")
    start_time_str = arrival.get("time")
    start_time = parse_time(start_time_str)

    candidates = []

    def build_prefix_steps(curr_location, curr_time, target_location, meeting_start):
        steps = []
        arrival_time = curr_time

        if curr_location != target_location:
            travel_duration = travel_lookup.get((curr_location, target_location))
            if travel_duration is None:
                return None, None
            arrival_time = curr_time + travel_duration
            steps.append({
                "action": "travel",
                "from": curr_location,
                "to": target_location,
                "duration_minutes": travel_duration,
                "arrival_time": format_time(arrival_time),
            })

        if arrival_time > meeting_start:
            return None, None

        if arrival_time < meeting_start:
            steps.append({
                "action": "wait",
                "location": target_location,
                "start_time": format_time(arrival_time),
                "end_time": format_time(meeting_start),
                "duration_minutes": meeting_start - arrival_time,
            })

        return steps, arrival_time

    def dfs(curr_location, curr_time, remaining_people, steps, met_any):
        if met_any:
            candidates.append({
                "start": {
                    "location": start_location,
                    "time": start_time_str,
                },
                "steps": deepcopy(steps),
            })

        for person in remaining_people:
            min_duration = min_meeting_by_person.get(person, 0)
            for slot in availability_by_person.get(person, []):
                target_location = slot["location"]
                slot_start = slot["start"]
                slot_end = slot["end"]

                if curr_location == target_location:
                    arrival_time = curr_time
                else:
                    travel_duration = travel_lookup.get((curr_location, target_location))
                    if travel_duration is None:
                        continue
                    arrival_time = curr_time + travel_duration

                earliest_start = max(arrival_time, slot_start)

                # Explore all feasible meeting starts that satisfy the minimum duration.
                # This guarantees complete search space coverage rather than assuming one start.
                latest_start = slot_end - min_duration
                if earliest_start > latest_start:
                    continue

                for meeting_start in range(earliest_start, latest_start + 1):
                    meeting_end = meeting_start + min_duration

                    prefix_steps, _ = build_prefix_steps(
                        curr_location, curr_time, target_location, meeting_start
                    )
                    if prefix_steps is None:
                        continue

                    next_steps = deepcopy(steps)
                    next_steps.extend(prefix_steps)
                    next_steps.append({
                        "action": "meet",
                        "person": person,
                        "location": target_location,
                        "start_time": format_time(meeting_start),
                        "end_time": format_time(meeting_end),
                        "duration_minutes": min_duration,
                    })

                    next_remaining = [p for p in remaining_people if p != person]
                    dfs(target_location, meeting_end, next_remaining, next_steps, True)

    dfs(start_location, start_time, people, [], False)

    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = repr(cand)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    return unique_candidates