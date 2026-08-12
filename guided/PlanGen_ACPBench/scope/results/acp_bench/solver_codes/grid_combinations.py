def combinations_func(data):
    from itertools import product

    def parse_pos(s):
        s = s.strip()
        if not (s.startswith("f") and s.endswith("f")):
            raise ValueError(f"Invalid position format: {s}")
        core = s[1:-1]
        a, b = core.split("-")
        return int(a), int(b)

    def fmt_pos(rc):
        return f"f{rc[0]}-{rc[1]}f"

    rows, cols = data["grid_size"]
    start = parse_pos(data["robot_start"])
    goal = parse_pos(data["robot_goal"])

    keys = data.get("keys", [])
    locked_positions = data.get("locked_positions", [])

    key_at = {}
    key_shape = {}
    for k in keys:
        pos = parse_pos(k["position"])
        key_at.setdefault(pos, []).append(k["id"])
        key_shape[k["id"]] = k["shape"]

    lock_shape_at = {}
    for lp in locked_positions:
        lock_shape_at[parse_pos(lp["position"])] = lp["shape"]

    def in_bounds(rc):
        return 0 <= rc[0] < rows and 0 <= rc[1] < cols

    def collected_shapes(collected_ids):
        return {key_shape[kid] for kid in collected_ids}

    def can_enter(rc, collected_ids):
        if rc not in lock_shape_at:
            return True
        return lock_shape_at[rc] in collected_shapes(collected_ids)

    def neighbors(rc):
        r, c = rc
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            nxt = (nr, nc)
            if in_bounds(nxt):
                yield nxt

    def actions_from_state(pos, collected_ids):
        acts = []

        # pickup actions
        for kid in key_at.get(pos, []):
            if kid not in collected_ids:
                acts.append(("pickup", kid))

        # move actions
        for nxt in neighbors(pos):
            if can_enter(nxt, collected_ids):
                acts.append(("move", nxt))

        return acts

    def apply_action(pos, collected_ids, action):
        kind, val = action
        if kind == "pickup":
            new_ids = frozenset(set(collected_ids) | {val})
            return pos, new_ids, f"pickup {val} {fmt_pos(pos)}"
        elif kind == "move":
            return val, collected_ids, f"move {fmt_pos(pos)} {fmt_pos(val)}"
        else:
            raise ValueError("Unknown action")

    # Small bounded exploration.
    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    max_steps = max(4, min(6, manhattan + 3))

    candidates = []
    seen_plans = set()

    frontier = [(start, frozenset(), [])]

    for _ in range(max_steps + 1):
        new_frontier = []
        for pos, collected_ids, plan in frontier:
            if pos == goal and plan:
                t = tuple(plan)
                if t not in seen_plans:
                    seen_plans.add(t)
                    candidates.append(plan)

            if len(plan) >= max_steps:
                continue

            for action in actions_from_state(pos, collected_ids):
                new_pos, new_ids, action_str = apply_action(pos, collected_ids, action)
                new_plan = plan + [action_str]
                new_frontier.append((new_pos, new_ids, new_plan))
        frontier = new_frontier

    # If still no candidates, try pure move combinations up to a short length.
    if not candidates:
        dirs = {
            "D": (1, 0),
            "U": (-1, 0),
            "R": (0, 1),
            "L": (0, -1),
        }
        for length in range(1, max_steps + 1):
            for seq in product(dirs.keys(), repeat=length):
                pos = start
                collected_ids = frozenset()
                plan = []
                ok = True
                for step in seq:
                    dr, dc = dirs[step]
                    nxt = (pos[0] + dr, pos[1] + dc)
                    if not in_bounds(nxt) or not can_enter(nxt, collected_ids):
                        ok = False
                        break
                    plan.append(f"move {fmt_pos(pos)} {fmt_pos(nxt)}")
                    pos = nxt
                    for kid in key_at.get(pos, []):
                        if kid not in collected_ids:
                            collected_ids = frozenset(set(collected_ids) | {kid})
                if ok and pos == goal:
                    t = tuple(plan)
                    if t not in seen_plans:
                        seen_plans.add(t)
                        candidates.append(plan)

    return candidates