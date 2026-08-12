def plan_func(data, constraints):
    def time_to_minutes(t):
        if t is None:
            return None
        if isinstance(t, (int, float)):
            return int(t)
        t = str(t).strip()
        if not t:
            return None
        parts = t.split(":")
        if len(parts) != 2:
            return None
        h, m = int(parts[0]), int(parts[1])
        return h * 60 + m

    def normalize_day(day):
        if day is None:
            return ""
        return str(day).strip().lower()

    day_order = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }

    def overlaps(s1, e1, s2, e2):
        return s1 < e2 and s2 < e1

    def get_plan_sort_key(plan):
        if not plan:
            return (10**9, 10**9, 10**9)
        first = plan[0]
        d = day_order.get(normalize_day(first.get("day")), 10**6)
        s = time_to_minutes(first.get("start"))
        e = time_to_minutes(first.get("end"))
        return (d, s if s is not None else 10**9, e if e is not None else 10**9)

    def extract_participants(plan, constraints):
        participants = set()

        # Explicit participants on segments
        for seg in plan:
            for key in ("participants", "attendees", "people"):
                vals = seg.get(key)
                if isinstance(vals, list):
                    participants.update(str(v) for v in vals)
                elif isinstance(vals, str):
                    participants.add(vals)

        # Explicit participants in constraints
        for key in ("participants", "attendees", "people", "required_participants"):
            vals = constraints.get(key)
            if isinstance(vals, list):
                participants.update(str(v) for v in vals)
            elif isinstance(vals, str):
                participants.add(vals)

        # Fallback: all people with schedules
        if not participants:
            participants.update(constraints.get("existing_schedules", {}).keys())

        return participants

    def parse_preferences(preferences):
        parsed = {
            "global": {
                "avoid_before": None,
                "avoid_after": None,
                "prefer_start_at_or_after": None,
                "prefer_end_at_or_before": None,
            },
            "by_person": {}
        }

        if not preferences:
            return parsed

        # Structured dict preferences
        if isinstance(preferences, dict):
            global_keys = (
                "avoid_before", "avoid_after",
                "prefer_start_at_or_after", "prefer_end_at_or_before",
                "not_before", "not_after",
                "would_rather_not_meet_before", "would_rather_not_meet_after",
            )
            for k in global_keys:
                if k in preferences:
                    val = time_to_minutes(preferences.get(k))
                    if k in ("avoid_before", "not_before", "would_rather_not_meet_before"):
                        parsed["global"]["avoid_before"] = val
                    elif k in ("avoid_after", "not_after", "would_rather_not_meet_after"):
                        parsed["global"]["avoid_after"] = val
                    elif k == "prefer_start_at_or_after":
                        parsed["global"]["prefer_start_at_or_after"] = val
                    elif k == "prefer_end_at_or_before":
                        parsed["global"]["prefer_end_at_or_before"] = val

            # Participant-specific preferences
            for person, pref in preferences.items():
                if isinstance(pref, dict):
                    parsed["by_person"].setdefault(str(person), {})
                    for k, v in pref.items():
                        val = time_to_minutes(v) if isinstance(v, (str, int, float)) else v
                        if k in ("avoid_before", "not_before", "would_rather_not_meet_before"):
                            parsed["by_person"][str(person)]["avoid_before"] = val
                        elif k in ("avoid_after", "not_after", "would_rather_not_meet_after"):
                            parsed["by_person"][str(person)]["avoid_after"] = val
                        elif k == "prefer_start_at_or_after":
                            parsed["by_person"][str(person)]["prefer_start_at_or_after"] = val
                        elif k == "prefer_end_at_or_before":
                            parsed["by_person"][str(person)]["prefer_end_at_or_before"] = val

        # List/string preferences are ignored unless structured enough to parse safely.
        return parsed

    work_hours = constraints.get("work_hours", {})
    work_start = time_to_minutes(work_hours.get("start")) if isinstance(work_hours, dict) else None
    work_end = time_to_minutes(work_hours.get("end")) if isinstance(work_hours, dict) else None

    existing = constraints.get("existing_schedules", {}) or {}
    parsed_prefs = parse_preferences(constraints.get("preferences"))

    valid_plans = []

    for plan in data:
        if not isinstance(plan, list) or not plan:
            continue

        participants = extract_participants(plan, constraints)
        ok = True

        for seg in plan:
            if not isinstance(seg, dict):
                ok = False
                break

            start = time_to_minutes(seg.get("start"))
            end = time_to_minutes(seg.get("end"))
            if start is None or end is None or start >= end:
                ok = False
                break

            # Work hours
            if work_start is not None and start < work_start:
                ok = False
                break
            if work_end is not None and end > work_end:
                ok = False
                break

            # Existing schedules
            for person in participants:
                busy_list = existing.get(person, [])
                for busy in busy_list:
                    b_start = time_to_minutes(busy.get("start"))
                    b_end = time_to_minutes(busy.get("end"))
                    if b_start is None or b_end is None:
                        continue
                    if overlaps(start, end, b_start, b_end):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

            # Global preferences
            gp = parsed_prefs["global"]
            if gp.get("avoid_before") is not None and start < gp["avoid_before"]:
                ok = False
                break
            if gp.get("avoid_after") is not None and end > gp["avoid_after"]:
                ok = False
                break
            if gp.get("prefer_start_at_or_after") is not None and start < gp["prefer_start_at_or_after"]:
                ok = False
                break
            if gp.get("prefer_end_at_or_before") is not None and end > gp["prefer_end_at_or_before"]:
                ok = False
                break

            # Participant-specific preferences
            for person in participants:
                pp = parsed_prefs["by_person"].get(person, {})
                if pp.get("avoid_before") is not None and start < pp["avoid_before"]:
                    ok = False
                    break
                if pp.get("avoid_after") is not None and end > pp["avoid_after"]:
                    ok = False
                    break
                if pp.get("prefer_start_at_or_after") is not None and start < pp["prefer_start_at_or_after"]:
                    ok = False
                    break
                if pp.get("prefer_end_at_or_before") is not None and end > pp["prefer_end_at_or_before"]:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            valid_plans.append(plan)

    if not valid_plans:
        return None

    valid_plans.sort(key=get_plan_sort_key)
    return valid_plans[0]