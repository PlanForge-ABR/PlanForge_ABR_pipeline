def plan_func(data, constraints):
    import re

    def get_stays(plan):
        return [x for x in plan if isinstance(x, dict) and "city" in x and "start_day" in x and "end_day" in x]

    def get_flights(plan):
        return [x for x in plan if isinstance(x, dict) and "from" in x and "to" in x and "flight_day" in x]

    def inclusive_duration(stay):
        return stay["end_day"] - stay["start_day"] + 1

    def plan_start_day(plan):
        stays = get_stays(plan)
        return min((s["start_day"] for s in stays), default=float("inf"))

    def plan_end_day(plan):
        stays = get_stays(plan)
        return max((s["end_day"] for s in stays), default=float("inf"))

    def city_stays_map(plan):
        m = {}
        for s in get_stays(plan):
            m.setdefault(s["city"], []).append(s)
        return m

    def stay_covers_window(stay, start_day, end_day):
        return stay["start_day"] <= start_day and stay["end_day"] >= end_day

    def stay_overlaps_window(stay, start_day, end_day):
        return not (stay["end_day"] < start_day or stay["start_day"] > end_day)

    def parse_time_to_minutes(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val * 60) if val <= 24 else int(val)
        if isinstance(val, str):
            m = re.search(r'(\d{1,2})(?::(\d{2}))?', val)
            if m:
                h = int(m.group(1))
                mm = int(m.group(2) or 0)
                return h * 60 + mm
        return None

    def extract_time_bounds_from_text(text):
        text = (text or "").lower()
        bounds = {}
        m = re.search(r'avoid meetings before\s+(\d{1,2}(?::\d{2})?)', text)
        if m:
            bounds["avoid_before"] = parse_time_to_minutes(m.group(1))
        m = re.search(r'avoid meetings after\s+(\d{1,2}(?::\d{2})?)', text)
        if m:
            bounds["avoid_after"] = parse_time_to_minutes(m.group(1))
        m = re.search(r'would rather not meet after\s+(\d{1,2}(?::\d{2})?)', text)
        if m:
            bounds["avoid_after"] = parse_time_to_minutes(m.group(1))
        m = re.search(r'would rather not meet before\s+(\d{1,2}(?::\d{2})?)', text)
        if m:
            bounds["avoid_before"] = parse_time_to_minutes(m.group(1))
        return bounds

    def check_stay_durations(plan, stay_durations):
        if not stay_durations:
            return True
        stays_by_city = city_stays_map(plan)
        for city, required in stay_durations.items():
            city_stays = stays_by_city.get(city, [])
            total = sum(inclusive_duration(s) for s in city_stays)
            if total != required:
                return False
        return True

    def check_time_window_constraints(plan, tw_constraints):
        if not tw_constraints:
            return True
        stays_by_city = city_stays_map(plan)
        for c in tw_constraints:
            city = c.get("city")
            start_day = c.get("start_day")
            end_day = c.get("end_day")
            mode = c.get("mode", "cover")
            if city is None or start_day is None or end_day is None:
                continue
            city_stays = stays_by_city.get(city, [])
            if mode == "overlap":
                ok = any(stay_overlaps_window(s, start_day, end_day) for s in city_stays)
            else:
                ok = any(stay_covers_window(s, start_day, end_day) for s in city_stays)
            if not ok:
                return False
        return True

    def check_transportation_constraint(plan, transportation_constraint):
        if not transportation_constraint:
            return True
        stays = sorted(get_stays(plan), key=lambda x: (x["start_day"], x["end_day"]))
        flights = get_flights(plan)
        between = transportation_constraint.get("between_cities")

        if between == "direct flights only":
            if len(stays) <= 1:
                return True
            for i in range(len(stays) - 1):
                a = stays[i]
                b = stays[i + 1]
                expected_day = a["end_day"]
                ok = any(
                    f.get("from") == a["city"] and
                    f.get("to") == b["city"] and
                    f.get("flight_day") == expected_day
                    for f in flights
                )
                if not ok:
                    return False
            return True

        return True

    def check_basic_plan_consistency(plan):
        stays = sorted(get_stays(plan), key=lambda x: (x["start_day"], x["end_day"]))
        flights = get_flights(plan)

        for s in stays:
            if s["end_day"] < s["start_day"]:
                return False

        for i in range(len(stays) - 1):
            a = stays[i]
            b = stays[i + 1]
            if b["start_day"] < a["start_day"]:
                return False
            if b["start_day"] != a["end_day"]:
                return False
            ok = any(
                f.get("from") == a["city"] and
                f.get("to") == b["city"] and
                f.get("flight_day") == a["end_day"]
                for f in flights
            )
            if not ok:
                return False
        return True

    def check_preference_item(plan, pref):
        stays = sorted(get_stays(plan), key=lambda x: (x["start_day"], x["end_day"]))
        stays_by_city = city_stays_map(plan)

        if isinstance(pref, str):
            bounds = extract_time_bounds_from_text(pref)
            # Candidate plans typically do not include meeting times; if a time-only preference
            # exists without meeting-time data, treat it as non-binding rather than failing.
            return True if bounds else True

        if not isinstance(pref, dict):
            return True

        ptype = pref.get("type")

        if ptype == "city_order":
            order = pref.get("order", [])
            actual = [s["city"] for s in stays]
            pos = -1
            for city in order:
                try:
                    idx = actual.index(city, pos + 1)
                except ValueError:
                    return False
                pos = idx
            return True

        if ptype == "required_cities":
            required = set(pref.get("cities", []))
            actual = {s["city"] for s in stays}
            return required.issubset(actual)

        if ptype == "forbidden_cities":
            forbidden = set(pref.get("cities", []))
            actual = {s["city"] for s in stays}
            return actual.isdisjoint(forbidden)

        if ptype == "stay":
            city = pref.get("city")
            min_days = pref.get("min_days")
            max_days = pref.get("max_days")
            exact_days = pref.get("exact_days")
            total = sum(inclusive_duration(s) for s in stays_by_city.get(city, []))
            if exact_days is not None and total != exact_days:
                return False
            if min_days is not None and total < min_days:
                return False
            if max_days is not None and total > max_days:
                return False
            return True

        if ptype == "trip":
            earliest_start = pref.get("earliest_start_day")
            latest_end = pref.get("latest_end_day")
            if earliest_start is not None and plan_start_day(plan) < earliest_start:
                return False
            if latest_end is not None and plan_end_day(plan) > latest_end:
                return False
            return True

        if ptype == "availability":
            city = pref.get("city")
            start_day = pref.get("start_day")
            end_day = pref.get("end_day")
            overlap_ok = pref.get("overlap_ok", False)
            city_stays = stays_by_city.get(city, [])
            if overlap_ok:
                return any(stay_overlaps_window(s, start_day, end_day) for s in city_stays)
            return any(stay_covers_window(s, start_day, end_day) for s in city_stays)

        if ptype == "meeting_time":
            # If meeting times are present in constraints and plan items, enforce them.
            # Otherwise, do not reject a plan solely because time-of-day data is absent.
            city = pref.get("city")
            event = pref.get("event")
            not_before = parse_time_to_minutes(pref.get("not_before"))
            not_after = parse_time_to_minutes(pref.get("not_after"))
            meetings = [
                x for x in plan
                if isinstance(x, dict) and x.get("type") == "meeting"
            ]
            relevant = [
                m for m in meetings
                if (city is None or m.get("city") == city) and (event is None or m.get("event") == event)
            ]
            if not relevant:
                return True
            for m in relevant:
                t = parse_time_to_minutes(m.get("time"))
                if t is None:
                    continue
                if not_before is not None and t < not_before:
                    return False
                if not_after is not None and t > not_after:
                    return False
            return True

        # Generic dict handling for common keys even without explicit type.
        city = pref.get("city")
        if city is not None and any(k in pref for k in ("min_days", "max_days", "exact_days")):
            total = sum(inclusive_duration(s) for s in stays_by_city.get(city, []))
            if "exact_days" in pref and total != pref["exact_days"]:
                return False
            if "min_days" in pref and total < pref["min_days"]:
                return False
            if "max_days" in pref and total > pref["max_days"]:
                return False

        if "start_day" in pref and "end_day" in pref and city is not None:
            overlap_ok = pref.get("overlap_ok", False)
            city_stays = stays_by_city.get(city, [])
            ok = any(
                stay_overlaps_window(s, pref["start_day"], pref["end_day"]) if overlap_ok
                else stay_covers_window(s, pref["start_day"], pref["end_day"])
                for s in city_stays
            )
            if not ok:
                return False

        return True

    def check_preferences(plan, preferences):
        if not preferences:
            return True

        if isinstance(preferences, dict):
            # Structured preference buckets
            if "city_order" in preferences:
                if not check_preference_item(plan, {"type": "city_order", "order": preferences["city_order"]}):
                    return False
            if "required_cities" in preferences:
                if not check_preference_item(plan, {"type": "required_cities", "cities": preferences["required_cities"]}):
                    return False
            if "forbidden_cities" in preferences:
                if not check_preference_item(plan, {"type": "forbidden_cities", "cities": preferences["forbidden_cities"]}):
                    return False
            if "trip" in preferences:
                trip_pref = dict(preferences["trip"])
                trip_pref["type"] = "trip"
                if not check_preference_item(plan, trip_pref):
                    return False
            if "stays" in preferences:
                for item in preferences["stays"]:
                    item = dict(item)
                    item.setdefault("type", "stay")
                    if not check_preference_item(plan, item):
                        return False
            if "availability" in preferences:
                av = preferences["availability"]
                if isinstance(av, list):
                    for item in av:
                        item = dict(item)
                        item.setdefault("type", "availability")
                        if not check_preference_item(plan, item):
                            return False
                elif isinstance(av, dict):
                    item = dict(av)
                    item.setdefault("type", "availability")
                    if not check_preference_item(plan, item):
                        return False
            if "meeting_time" in preferences:
                mt = preferences["meeting_time"]
                if isinstance(mt, list):
                    for item in mt:
                        item = dict(item)
                        item.setdefault("type", "meeting_time")
                        if not check_preference_item(plan, item):
                            return False
                elif isinstance(mt, dict):
                    item = dict(mt)
                    item.setdefault("type", "meeting_time")
                    if not check_preference_item(plan, item):
                        return False
            if "notes" in preferences:
                notes = preferences["notes"]
                if isinstance(notes, str):
                    if not check_preference_item(plan, notes):
                        return False
                elif isinstance(notes, list):
                    for note in notes:
                        if not check_preference_item(plan, note):
                            return False
            # Also evaluate any list-like generic entries
            for k, v in preferences.items():
                if k in {"city_order", "required_cities", "forbidden_cities", "trip", "stays", "availability", "meeting_time", "notes"}:
                    continue
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) or isinstance(item, str):
                            if not check_preference_item(plan, item):
                                return False
                elif isinstance(v, dict):
                    if not check_preference_item(plan, v):
                        return False
                elif isinstance(v, str):
                    if not check_preference_item(plan, v):
                        return False
            return True

        if isinstance(preferences, list):
            return all(check_preference_item(plan, p) for p in preferences)

        if isinstance(preferences, str):
            return check_preference_item(plan, preferences)

        return True

    valid_plans = []
    for plan in data:
        if not isinstance(plan, list):
            continue
        if not check_basic_plan_consistency(plan):
            continue
        if not check_stay_durations(plan, constraints.get("stay_durations")):
            continue
        if not check_time_window_constraints(plan, constraints.get("time_window_constraints")):
            continue
        if not check_transportation_constraint(plan, constraints.get("transportation_constraint")):
            continue
        if not check_preferences(plan, constraints.get("preferences")):
            continue
        valid_plans.append(plan)

    if not valid_plans:
        return None

    valid_plans.sort(key=lambda p: (plan_start_day(p), plan_end_day(p), len(get_stays(p))))
    return valid_plans[0]