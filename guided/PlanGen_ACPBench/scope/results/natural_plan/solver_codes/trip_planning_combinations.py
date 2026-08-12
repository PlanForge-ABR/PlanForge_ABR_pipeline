def combinations_func(data):
    from itertools import permutations, product

    cities = list(data.get("cities_to_visit", []))
    total_days = data.get("total_days")
    total_cities = data.get("total_cities", len(cities))
    direct_flights = data.get("direct_flights", [])
    stay_durations = data.get("stay_durations", {}) or {}
    time_window_constraints = data.get("time_window_constraints", []) or []
    transportation_constraint = data.get("transportation_constraint", {}) or {}

    if total_cities is not None and len(cities) != total_cities:
        # Keep only consistent inputs; if inconsistent, still search over provided cities.
        pass

    # Build allowed directed flight edges.
    allowed_edges = set()
    for pair in direct_flights:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            allowed_edges.add((pair[0], pair[1]))

    direct_only = transportation_constraint.get("between_cities") == "direct flights only"

    def compositions(total, parts, minimums=None):
        """Generate all integer vectors of length `parts` summing to `total` with lower bounds."""
        if minimums is None:
            minimums = [1] * parts
        minimum_sum = sum(minimums)
        if total < minimum_sum:
            return
        remainder = total - minimum_sum

        def rec(i, left, acc):
            if i == parts - 1:
                yield acc + [minimums[i] + left]
                return
            for x in range(left + 1):
                yield from rec(i + 1, left - x, acc + [minimums[i] + x])

        yield from rec(0, remainder, [])

    def duration_assignments_for_order(order):
        """
        Generate duration dictionaries.
        If all cities have specified durations, use them exactly.
        Otherwise, exhaustively allocate positive durations so that the itinerary
        spans total_days under the inclusive-day flight convention:
        total_days = sum(city_stays) - (number_of_cities - 1)
        """
        if all(city in stay_durations for city in order):
            durations = {city: stay_durations[city] for city in order}
            if total_days is None or sum(durations.values()) - (len(order) - 1) == total_days:
                yield durations
            return

        if total_days is None:
            return

        mins = [stay_durations.get(city, 1) for city in order]
        target_sum = total_days + (len(order) - 1)
        for vals in compositions(target_sum, len(order), mins):
            yield {city: vals[i] for i, city in enumerate(order)}

    def build_plan(order, durations):
        plan = []
        current_start = 1
        city_ranges = {}

        for i, city in enumerate(order):
            dur = durations[city]
            end_day = current_start + dur - 1
            city_ranges[city] = (current_start, end_day)
            plan.append({
                "city": city,
                "start_day": current_start,
                "end_day": end_day
            })

            if i < len(order) - 1:
                next_city = order[i + 1]
                flight_day = end_day
                plan.append({
                    "flight_day": flight_day,
                    "from": city,
                    "to": next_city
                })
                current_start = flight_day

        return plan, city_ranges

    def satisfies_flights(order):
        if not direct_only:
            return True
        for a, b in zip(order, order[1:]):
            if (a, b) not in allowed_edges:
                return False
        return True

    def satisfies_time_windows(city_ranges):
        for constraint in time_window_constraints:
            city = constraint.get("city")
            if city not in city_ranges:
                return False
            stay_start, stay_end = city_ranges[city]
            window_start = constraint.get("start_day")
            window_end = constraint.get("end_day")

            # Require at least one day of overlap with the requested window.
            if window_start is not None and stay_end < window_start:
                return False
            if window_end is not None and stay_start > window_end:
                return False
        return True

    candidates = []

    for order in permutations(cities):
        if not satisfies_flights(order):
            continue

        for durations in duration_assignments_for_order(order):
            plan, city_ranges = build_plan(order, durations)

            # Validate total trip length if provided.
            if total_days is not None:
                actual_total_days = max(seg["end_day"] for seg in plan if "city" in seg)
                if actual_total_days != total_days:
                    continue

            if not satisfies_time_windows(city_ranges):
                continue

            candidates.append(plan)

    return candidates