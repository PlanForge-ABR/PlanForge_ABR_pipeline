def combinations_func(data):
    from itertools import product

    def rover_from_store(store):
        digits = "".join(ch for ch in str(store) if ch.isdigit())
        return f"rover{digits}" if digits else "rover1"

    target_stores = data.get("target_store_empty", [])
    if not target_stores:
        return []

    plans = []

    objectives = ["objective0", "objective1"]
    modes = ["colour", "high_res", "low_res"]
    waypoints = [f"waypoint{i}" for i in range(10)]

    for store in target_stores:
        rover = rover_from_store(store)

        # Core single-step candidate: empty the store directly.
        plans.append([f"drop {rover} {store}"])

        # Two-step candidates: drop, then communicate some data.
        for objective, mode, wp_from, wp_to in product(objectives, modes, waypoints, waypoints):
            if wp_from == wp_to:
                continue
            plans.append([
                f"drop {rover} {store}",
                f"communicate_image_data {rover} general {objective} {mode} {wp_from} {wp_to}",
            ])

        # Small additional variants with repeated drop attempts kept short.
        for wp_from, wp_to in product(waypoints[:3], waypoints[3:6]):
            plans.append([
                f"drop {rover} {store}",
                f"drop {rover} {store}",
                f"communicate_image_data {rover} general objective0 colour {wp_from} {wp_to}",
            ])

    return plans