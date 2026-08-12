def combinations_func(data):
    from itertools import permutations, product

    targets = data.get("target_truck_locations", [])
    if not targets:
        return []

    # Collect trucks and destination depots from the input.
    trucks = []
    dest_by_truck = {}
    depots = {"depot0"}
    for item in targets:
        truck = item["truck"]
        loc = item["location"]
        trucks.append(truck)
        dest_by_truck[truck] = loc
        depots.add(loc)

    depots = sorted(depots)

    # For this domain, assume trucks start at depot0 unless already there.
    base_actions = []
    for truck in trucks:
        dest = dest_by_truck[truck]
        if dest != "depot0":
            base_actions.append(f"drive {truck} depot0 {dest}")

    candidates = []

    # 1) Direct plans: all orderings of the independent drive actions.
    if base_actions:
        for perm in permutations(base_actions):
            candidates.append(list(perm))
    else:
        candidates.append([])

    # 2) Small exploratory variants with one intermediate depot for a truck.
    #    This keeps plans short while still generating candidate sequences.
    for truck in trucks:
        dest = dest_by_truck[truck]
        if dest == "depot0":
            continue

        intermediates = [d for d in depots if d not in ("depot0", dest)]
        for mid in intermediates:
            modified_actions = []
            for t in trucks:
                if t == truck:
                    modified_actions.append(f"drive {t} depot0 {mid}")
                    modified_actions.append(f"drive {t} {mid} {dest}")
                else:
                    other_dest = dest_by_truck[t]
                    if other_dest != "depot0":
                        modified_actions.append(f"drive {t} depot0 {other_dest}")

            for perm in permutations(modified_actions):
                candidates.append(list(perm))

    # 3) Deduplicate while preserving order.
    seen = set()
    unique_candidates = []
    for plan in candidates:
        key = tuple(plan)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(plan)

    return unique_candidates