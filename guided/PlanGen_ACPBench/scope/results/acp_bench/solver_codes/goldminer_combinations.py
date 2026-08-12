def combinations_func(data):
    from itertools import product

    goal_pos = data.get("goal_robot_position")
    goal_hold = data.get("goal_holding")

    # Known object locations inferred from the exemplar.
    item_locations = {
        "gold": "f3-4f",
    }

    def neighbors(loc):
        # Lightweight location variation generator for candidate move sources.
        # For strings like "f3-3f", vary the middle number by +/- 1.
        out = []
        try:
            if "-" in loc and loc.endswith("f"):
                left, right = loc.split("-")
                if right[:-1].isdigit():
                    n = int(right[:-1])
                    for d in (-1, 1):
                        m = n + d
                        if m >= 0:
                            out.append(f"{left}-{m}f")
        except Exception:
            pass
        return out

    candidates = []
    seen = set()

    def add(plan):
        key = tuple(plan)
        if key not in seen:
            seen.add(key)
            candidates.append(plan)

    item_loc = item_locations.get(goal_hold) if goal_hold else None

    # Build candidate sequences up to length 3.
    if goal_hold and item_loc and goal_pos:
        # Canonical plan from the exemplar pattern: pick, then move to goal.
        add([f"pick-{goal_hold} {item_loc}", f"move {item_loc} {goal_pos}"])

        # If already at the goal position and item location matches, picking alone is a candidate.
        if item_loc == goal_pos:
            add([f"pick-{goal_hold} {item_loc}"])

        # Explore short variants using nearby source/target positions.
        move_sources = [item_loc] + neighbors(item_loc)
        move_targets = [goal_pos] + neighbors(goal_pos)

        for src, tgt in product(move_sources, move_targets):
            # Move to item, pick, then move to goal.
            if src != item_loc and tgt == goal_pos:
                add([f"move {src} {item_loc}", f"pick-{goal_hold} {item_loc}", f"move {item_loc} {goal_pos}"])
            # Pick at item, then move to a plausible target.
            if tgt != item_loc:
                add([f"pick-{goal_hold} {item_loc}", f"move {item_loc} {tgt}"])

    elif goal_hold and item_loc:
        # Holding goal only.
        add([f"pick-{goal_hold} {item_loc}"])
        for src in neighbors(item_loc):
            add([f"move {src} {item_loc}", f"pick-{goal_hold} {item_loc}"])

    elif goal_pos:
        # Position goal only.
        for src in neighbors(goal_pos):
            add([f"move {src} {goal_pos}"])

    return candidates