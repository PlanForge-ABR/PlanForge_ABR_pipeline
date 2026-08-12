def combinations_func(data):
    from itertools import product

    def infer_grid(n):
        pairs = []
        for r in range(1, int(n ** 0.5) + 1):
            if n % r == 0:
                c = n // r
                pairs.append((r, c))
        if not pairs:
            return 1, n
        pairs.sort(key=lambda rc: (abs(rc[0] - rc[1]), -rc[0]))
        return pairs[0]

    def tile_num(tile):
        return int(str(tile).split("_")[1])

    def tile_name(num):
        return f"tile_{num}"

    rows, cols = infer_grid(data["tiles"])

    def neighbors(tile):
        n = tile_num(tile)
        r = (n - 1) // cols
        c = (n - 1) % cols
        out = []
        if c > 0:
            out.append(("left", tile_name(n - 1)))
        if c < cols - 1 and n + 1 <= data["tiles"]:
            out.append(("right", tile_name(n + 1)))
        if r > 0:
            out.append(("down", tile_name(n - cols)))
        if r < rows - 1 and n + cols <= data["tiles"]:
            out.append(("up", tile_name(n + cols)))
        return out

    robots = sorted(data["robot_positions"].keys())
    init_positions = dict(data["robot_positions"])
    goal = data.get("goal", {})

    def occupied_map(state):
        return {pos: robot for robot, pos in state.items()}

    def valid_actions(state):
        occ = occupied_map(state)
        acts = []
        for robot in robots:
            src = state[robot]
            for direction, dst in neighbors(src):
                if dst not in occ:
                    acts.append(f"{direction} {robot} {src} {dst}")
        return acts

    def apply_action(state, action):
        parts = action.split()
        if len(parts) != 4:
            return None
        direction, robot, src, dst = parts
        if robot not in state or state[robot] != src:
            return None
        legal = {a.split()[3]: a.split()[0] for a in valid_actions(state) if a.split()[1] == robot and a.split()[2] == src}
        if dst not in legal or legal[dst] != direction:
            return None
        new_state = dict(state)
        new_state[robot] = dst
        return new_state

    def manhattan(tile_a, tile_b):
        a = tile_num(tile_a) - 1
        b = tile_num(tile_b) - 1
        ar, ac = divmod(a, cols)
        br, bc = divmod(b, cols)
        return abs(ar - br) + abs(ac - bc)

    def score_plan(state, plan):
        score = 0
        for k, v in goal.items():
            if k.endswith("_at"):
                robot = k[:-3]
                if robot in state:
                    d0 = manhattan(init_positions[robot], v)
                    d1 = manhattan(state[robot], v)
                    score += (d0 - d1) * 3
                    if state[robot] == v:
                        score += 20
            elif k.endswith("_clear") and v is True:
                tile = k[:-6]
                initially_occupied = any(pos == tile for pos in init_positions.values())
                finally_occupied = any(pos == tile for pos in state.values())
                if initially_occupied and not finally_occupied:
                    score += 15
                elif not finally_occupied:
                    score += 3
        score -= max(0, len(plan) - 1)
        return score

    max_len = 4
    candidates = []

    frontier = [(init_positions, [])]
    for _ in range(max_len):
        new_frontier = []
        for state, plan in frontier:
            acts = valid_actions(state)
            for act in acts:
                next_state = apply_action(state, act)
                if next_state is None:
                    continue
                next_plan = plan + [act]
                candidates.append((score_plan(next_state, next_plan), next_plan))
                new_frontier.append((next_state, next_plan))
        frontier = new_frontier

    seen = set()
    unique = []
    for sc, plan in sorted(candidates, key=lambda x: (-x[0], len(x[1]), x[1])):
        key = tuple(plan)
        if key not in seen:
            seen.add(key)
            unique.append(plan)

    return unique