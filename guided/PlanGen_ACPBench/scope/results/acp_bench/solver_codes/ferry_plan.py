def plan_func(data, constraints):
    locations = set(constraints.get("locations", []))
    cars = set(constraints.get("cars", []))
    capacity = constraints.get("ferry_capacity", 1)

    initial_ferry_location = constraints.get("initial_ferry_location")
    initial_ferry_empty = constraints.get("initial_ferry_empty", True)
    initial_car_locations = dict(constraints.get("initial_car_locations", {}))

    def parse_action(action):
        if not isinstance(action, str):
            return None
        parts = action.strip().split()
        if not parts:
            return None
        return parts

    def is_valid_plan(plan):
        if not isinstance(plan, list):
            return False

        ferry_location = initial_ferry_location
        ferry_load = []

        if not initial_ferry_empty:
            return False

        car_locations = dict(initial_car_locations)

        # Basic consistency checks on initial state
        if ferry_location not in locations:
            return False
        for car, loc in car_locations.items():
            if car not in cars or loc not in locations:
                return False

        for action in plan:
            parts = parse_action(action)
            if parts is None:
                return False

            op = parts[0]

            if op == "board":
                if len(parts) != 3:
                    return False
                car, loc = parts[1], parts[2]

                if car not in cars or loc not in locations:
                    return False
                if ferry_location != loc:
                    return False
                if len(ferry_load) >= capacity:
                    return False
                if car_locations.get(car) != loc:
                    return False

                ferry_load.append(car)
                del car_locations[car]

            elif op == "debark":
                if len(parts) != 3:
                    return False
                car, loc = parts[1], parts[2]

                if car not in cars or loc not in locations:
                    return False
                if ferry_location != loc:
                    return False
                if car not in ferry_load:
                    return False

                ferry_load.remove(car)
                car_locations[car] = loc

            elif op == "sail":
                if len(parts) != 3:
                    return False
                src, dst = parts[1], parts[2]

                if src not in locations or dst not in locations:
                    return False
                if ferry_location != src:
                    return False
                if src == dst:
                    return False

                ferry_location = dst

            else:
                return False

        return True

    for plan in data:
        if is_valid_plan(plan):
            return plan

    return None