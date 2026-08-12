def plan_func(data, constraints):
    def build_object_sets(objects):
        sets = {}
        for obj_type, count in (objects or {}).items():
            base = obj_type[:-1] if obj_type.endswith("s") else obj_type
            sets[obj_type] = {f"{base}{i}" for i in range(count)}
        return sets

    def parse_initial_truck_locations(initial_locs):
        truck_locations = {}
        for item in initial_locs or []:
            truck = item.get("truck")
            location = item.get("location")
            if truck is not None and location is not None:
                truck_locations[truck] = location
        return truck_locations

    def is_valid_drive(parts, truck_locations, object_sets):
        if len(parts) != 4:
            return False
        _, truck, src, dst = parts

        valid_trucks = object_sets.get("trucks", set())
        valid_depots = object_sets.get("depots", set())

        if valid_trucks and truck not in valid_trucks:
            return False
        if valid_depots and (src not in valid_depots or dst not in valid_depots):
            return False
        if truck not in truck_locations:
            return False
        if truck_locations[truck] != src:
            return False

        truck_locations[truck] = dst
        return True

    def is_valid_plan(plan, constraints):
        if not isinstance(plan, list):
            return False

        object_sets = build_object_sets(constraints.get("objects", {}))
        truck_locations = parse_initial_truck_locations(
            constraints.get("initial_truck_locations", [])
        )

        for step in plan:
            if not isinstance(step, str):
                return False
            parts = step.strip().split()
            if not parts:
                return False

            action = parts[0].lower()
            if action == "drive":
                if not is_valid_drive(parts, truck_locations, object_sets):
                    return False
            else:
                return False

        return True

    valid_plans = []
    for plan in data:
        if is_valid_plan(plan, constraints):
            valid_plans.append(plan)

    if not valid_plans:
        return None

    # Prefer shortest valid plan; break ties lexicographically.
    return min(valid_plans, key=lambda p: (len(p), tuple(p)))