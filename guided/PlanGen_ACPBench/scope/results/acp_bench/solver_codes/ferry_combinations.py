def combinations_func(data):
    from itertools import product

    target_ferry_location = data.get("target_ferry_location")
    target_car_on_ferry = data.get("target_car_on_ferry")

    default_locations = [f"l{i}" for i in range(10)]

    locations = data.get("locations", default_locations)
    if target_ferry_location is not None and target_ferry_location not in locations:
        locations = list(locations) + [target_ferry_location]

    candidates = []

    origin_locations = [loc for loc in locations if loc != target_ferry_location]
    for origin in origin_locations:
        plan = [
            f"board {target_car_on_ferry} {origin}",
            f"sail {origin} {target_ferry_location}",
        ]
        candidates.append(plan)

    if target_ferry_location is not None:
        candidates.append([
            f"board {target_car_on_ferry} {target_ferry_location}",
            f"sail {target_ferry_location} {target_ferry_location}",
        ])

    return candidates