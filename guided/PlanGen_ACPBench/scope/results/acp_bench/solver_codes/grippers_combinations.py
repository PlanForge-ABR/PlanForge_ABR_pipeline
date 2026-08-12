def combinations_func(data):
    from itertools import product

    robots = data.get("robots", [])
    target_rooms = data.get("target_rooms", [])

    def room_sort_key(r):
        digits = "".join(ch for ch in str(r) if ch.isdigit())
        return (str(r).rstrip(digits), int(digits) if digits else 0, str(r))

    def expand_rooms(targets):
        rooms = set(targets)
        for room in targets:
            s = str(room)
            prefix = "".join(ch for ch in s if not ch.isdigit())
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits:
                n = int(digits)
                if n > 1:
                    rooms.add(f"{prefix}{n-1}")
                rooms.add(f"{prefix}{n+1}")
        return sorted(rooms, key=room_sort_key)

    candidate_rooms = expand_rooms(target_rooms)
    plans = []

    # 1-step plans: direct move to target from any candidate source room
    for robot, target in product(robots, target_rooms):
        for src in candidate_rooms:
            plans.append([f"move {robot} {src} {target}"])

    # 2-step plans: optional self-move/staging move, then move to target
    for robot, target in product(robots, target_rooms):
        for mid in candidate_rooms:
            plans.append([
                f"move {robot} {mid} {mid}",
                f"move {robot} {mid} {target}",
            ])
            for src in candidate_rooms:
                plans.append([
                    f"move {robot} {src} {mid}",
                    f"move {robot} {mid} {target}",
                ])

    # 3-step plans: two transitions before reaching target
    for robot, target in product(robots, target_rooms):
        for src, mid1, mid2 in product(candidate_rooms, repeat=3):
            plans.append([
                f"move {robot} {src} {mid1}",
                f"move {robot} {mid1} {mid2}",
                f"move {robot} {mid2} {target}",
            ])

    # Deduplicate while preserving order
    unique_plans = []
    seen = set()
    for plan in plans:
        key = tuple(plan)
        if key not in seen:
            seen.add(key)
            unique_plans.append(plan)

    return unique_plans