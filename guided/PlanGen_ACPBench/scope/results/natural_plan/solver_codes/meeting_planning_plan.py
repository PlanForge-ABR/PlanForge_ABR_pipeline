def plan_func(data, constraints):
    import re

    def parse_time_to_minutes(t):
        if t is None:
            return None
        if isinstance(t, (int, float)):
            return int(t)
        s = str(t).strip()
        m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*([AaPp][Mm])\s*$", s)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            ap = m.group(3).upper()
            if hh == 12:
                hh = 0
            total = hh * 60 + mm
            if ap == "PM":
                total += 12 * 60
            return total
        m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", s)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        return None

    def get_plan_start_minutes(plan):
        start = plan.get("start", {})
        t = parse_time_to_minutes(start.get("time"))
        if t is not None:
            return t
        first_times = []
        for step in plan.get("steps", []):
            for key in ("start_time", "arrival_time", "end_time"):
                val = parse_time_to_minutes(step.get(key))
                if val is not None:
                    first_times.append(val)
        return min(first_times) if first_times else float("inf")

    def get_plan_end_minutes(plan):
        vals = []
        for step in plan.get("steps", []):
            for key in ("end_time", "arrival_time", "start_time"):
                val = parse_time_to_minutes(step.get(key))
                if val is not None:
                    vals.append(val)
        return max(vals) if vals else get_plan_start_minutes(plan)

    def collect_meetings(plan):
        meetings = []
        for step in plan.get("steps", []):
            if step.get("action") == "meet":
                meetings.append({
                    "person": step.get("person"),
                    "location": step.get("location"),
                    "start": parse_time_to_minutes(step.get("start_time")),
                    "end": parse_time_to_minutes(step.get("end_time")),
                    "duration": step.get("duration_minutes")
                })
        return meetings

    def collect_travels(plan):
        return [step for step in plan.get("steps", []) if step.get("action") == "travel"]

    def collect_waits(plan):
        return [step for step in plan.get("steps", []) if step.get("action") == "wait"]

    def check_arrival(plan, arrival_constraint):
        if not arrival_constraint:
            return True
        start = plan.get("start", {})
        if "location" in arrival_constraint and start.get("location") != arrival_constraint.get("location"):
            return False
        if "time" in arrival_constraint:
            if parse_time_to_minutes(start.get("time")) != parse_time_to_minutes(arrival_constraint.get("time")):
                return False
        return True

    def check_availability(plan, availability_list):
        if not availability_list:
            return True
        meetings = collect_meetings(plan)
        for avail in availability_list:
            person = avail.get("person")
            req_loc = avail.get("location")
            win_start = parse_time_to_minutes(avail.get("start_time"))
            win_end = parse_time_to_minutes(avail.get("end_time"))
            person_meetings = [m for m in meetings if m.get("person") == person]
            if not person_meetings:
                return False
            for m in person_meetings:
                if req_loc is not None and m.get("location") != req_loc:
                    return False
                if win_start is not None and (m.get("start") is None or m.get("start") < win_start):
                    return False
                if win_end is not None and (m.get("end") is None or m.get("end") > win_end):
                    return False
        return True

    def check_minimum_meeting_times(plan, minimums):
        if not minimums:
            return True
        meetings = collect_meetings(plan)
        totals = {}
        for m in meetings:
            person = m.get("person")
            dur = m.get("duration")
            if dur is None and m.get("start") is not None and m.get("end") is not None:
                dur = m["end"] - m["start"]
            totals[person] = totals.get(person, 0) + (dur or 0)
        for req in minimums:
            if totals.get(req.get("person"), 0) < req.get("minutes", 0):
                return False
        return True

    def check_step_consistency(plan):
        steps = plan.get("steps", [])
        prev_location = plan.get("start", {}).get("location")
        prev_time = parse_time_to_minutes(plan.get("start", {}).get("time"))

        for step in steps:
            action = step.get("action")

            if action == "travel":
                if prev_location is not None and step.get("from") is not None and step.get("from") != prev_location:
                    return False
                arr = parse_time_to_minutes(step.get("arrival_time"))
                dur = step.get("duration_minutes")
                if prev_time is not None and arr is not None and dur is not None:
                    if prev_time + int(dur) != arr:
                        return False
                prev_location = step.get("to")
                if arr is not None:
                    prev_time = arr

            elif action == "wait":
                loc = step.get("location")
                st = parse_time_to_minutes(step.get("start_time"))
                en = parse_time_to_minutes(step.get("end_time"))
                dur = step.get("duration_minutes")
                if prev_location is not None and loc is not None and loc != prev_location:
                    return False
                if prev_time is not None and st is not None and st != prev_time:
                    return False
                if st is not None and en is not None and dur is not None:
                    if st + int(dur) != en:
                        return False
                if en is not None:
                    prev_time = en
                prev_location = loc

            elif action == "meet":
                loc = step.get("location")
                st = parse_time_to_minutes(step.get("start_time"))
                en = parse_time_to_minutes(step.get("end_time"))
                dur = step.get("duration_minutes")
                if prev_location is not None and loc is not None and loc != prev_location:
                    return False
                if prev_time is not None and st is not None and st != prev_time:
                    return False
                if st is not None and en is not None and dur is not None:
                    if st + int(dur) != en:
                        return False
                if en is not None:
                    prev_time = en
                prev_location = loc

            else:
                st = parse_time_to_minutes(step.get("start_time"))
                en = parse_time_to_minutes(step.get("end_time"))
                if prev_time is not None and st is not None and st != prev_time:
                    return False
                if en is not None:
                    prev_time = en

        return True

    def check_preferences(plan, preferences):
        if not preferences:
            return True

        meetings = collect_meetings(plan)
        travels = collect_travels(plan)
        waits = collect_waits(plan)
        plan_start = get_plan_start_minutes(plan)
        plan_end = get_plan_end_minutes(plan)

        if isinstance(preferences, dict):
            pref_items = [preferences]
        elif isinstance(preferences, list):
            pref_items = preferences
        else:
            pref_items = [preferences]

        for pref in pref_items:
            if isinstance(pref, str):
                text = pref.lower()

                m = re.search(r"avoid meetings before\s+(\d{1,2}:\d{2}(?:\s*[ap]m)?)", text)
                if m:
                    cutoff = parse_time_to_minutes(m.group(1).upper().replace(" ", ""))
                    for mt in meetings:
                        if mt.get("start") is not None and mt["start"] < cutoff:
                            return False

                m = re.search(r"would rather not meet after\s+(\d{1,2}:\d{2}(?:\s*[ap]m)?)", text)
                if m:
                    cutoff = parse_time_to_minutes(m.group(1).upper().replace(" ", ""))
                    for mt in meetings:
                        if mt.get("end") is not None and mt["end"] > cutoff:
                            return False

                m = re.search(r"avoid meetings after\s+(\d{1,2}:\d{2}(?:\s*[ap]m)?)", text)
                if m:
                    cutoff = parse_time_to_minutes(m.group(1).upper().replace(" ", ""))
                    for mt in meetings:
                        if mt.get("start") is not None and mt["start"] >= cutoff:
                            return False

                continue

            if not isinstance(pref, dict):
                continue

            person = pref.get("person")

            if "avoid_meetings_before" in pref:
                cutoff = parse_time_to_minutes(pref.get("avoid_meetings_before"))
                for mt in meetings:
                    if (person is None or mt.get("person") == person) and mt.get("start") is not None and mt["start"] < cutoff:
                        return False

            if "avoid_meetings_after" in pref:
                cutoff = parse_time_to_minutes(pref.get("avoid_meetings_after"))
                for mt in meetings:
                    if (person is None or mt.get("person") == person) and mt.get("start") is not None and mt["start"] >= cutoff:
                        return False

            if "would_rather_not_meet_after" in pref:
                cutoff = parse_time_to_minutes(pref.get("would_rather_not_meet_after"))
                for mt in meetings:
                    if (person is None or mt.get("person") == person) and mt.get("end") is not None and mt["end"] > cutoff:
                        return False

            if "avoid_locations" in pref:
                avoid_locations = set(pref.get("avoid_locations") or [])
                for step in plan.get("steps", []):
                    loc = step.get("location")
                    if loc in avoid_locations:
                        return False
                    if step.get("action") == "travel" and (step.get("from") in avoid_locations or step.get("to") in avoid_locations):
                        return False

            if "required_locations" in pref:
                required_locations = set(pref.get("required_locations") or [])
                visited = set()
                start_loc = plan.get("start", {}).get("location")
                if start_loc:
                    visited.add(start_loc)
                for step in plan.get("steps", []):
                    for key in ("location", "from", "to"):
                        if step.get(key):
                            visited.add(step.get(key))
                if not required_locations.issubset(visited):
                    return False

            if "max_total_travel_minutes" in pref:
                total_travel = sum(int(t.get("duration_minutes", 0) or 0) for t in travels)
                if total_travel > int(pref.get("max_total_travel_minutes")):
                    return False

            if "max_total_wait_minutes" in pref:
                total_wait = sum(int(w.get("duration_minutes", 0) or 0) for w in waits)
                if total_wait > int(pref.get("max_total_wait_minutes")):
                    return False

            if "plan_must_end_by" in pref:
                cutoff = parse_time_to_minutes(pref.get("plan_must_end_by"))
                if plan_end > cutoff:
                    return False

            if "plan_must_start_after" in pref:
                cutoff = parse_time_to_minutes(pref.get("plan_must_start_after"))
                if plan_start < cutoff:
                    return False

            if "min_city_stay_minutes" in pref:
                city = pref.get("city") or pref.get("location")
                mins = int(pref.get("min_city_stay_minutes"))
                total = 0
                for step in plan.get("steps", []):
                    if step.get("action") in ("wait", "meet") and step.get("location") == city:
                        dur = step.get("duration_minutes")
                        if dur is None:
                            st = parse_time_to_minutes(step.get("start_time"))
                            en = parse_time_to_minutes(step.get("end_time"))
                            dur = (en - st) if st is not None and en is not None else 0
                        total += int(dur or 0)
                if total < mins:
                    return False

            if "max_city_stay_minutes" in pref:
                city = pref.get("city") or pref.get("location")
                mins = int(pref.get("max_city_stay_minutes"))
                total = 0
                for step in plan.get("steps", []):
                    if step.get("action") in ("wait", "meet") and step.get("location") == city:
                        dur = step.get("duration_minutes")
                        if dur is None:
                            st = parse_time_to_minutes(step.get("start_time"))
                            en = parse_time_to_minutes(step.get("end_time"))
                            dur = (en - st) if st is not None and en is not None else 0
                        total += int(dur or 0)
                if total > mins:
                    return False

            if "participant_availability_limits" in pref:
                limits = pref.get("participant_availability_limits") or []
                for lim in limits:
                    lim_person = lim.get("person")
                    for mt in meetings:
                        if lim_person is not None and mt.get("person") != lim_person:
                            continue
                        if "not_before" in lim:
                            cutoff = parse_time_to_minutes(lim.get("not_before"))
                            if mt.get("start") is not None and mt["start"] < cutoff:
                                return False
                        if "not_after" in lim:
                            cutoff = parse_time_to_minutes(lim.get("not_after"))
                            if mt.get("end") is not None and mt["end"] > cutoff:
                                return False

        return True

    valid_plans = []
    for plan in data:
        if not check_step_consistency(plan):
            continue
        if not check_arrival(plan, constraints.get("arrival")):
            continue
        if not check_availability(plan, constraints.get("availability")):
            continue
        if not check_minimum_meeting_times(plan, constraints.get("minimum_meeting_times_minutes")):
            continue
        if not check_preferences(plan, constraints.get("preferences")):
            continue
        valid_plans.append(plan)

    if not valid_plans:
        return None

    valid_plans.sort(key=lambda p: (get_plan_start_minutes(p), get_plan_end_minutes(p)))
    return valid_plans[:1]