def combinations_func(data):
    from itertools import product

    width = data["grid_size"]["width"]
    height = data["grid_size"]["height"]
    start = data["robot_current_location"]
    target = data["target_location"]
    blocked = set(data.get("unavailable_cells", []))
    visited = set(data.get("visited_cells", []))

    def parse_loc(loc):
        parts = loc.split("-")
        x = int(parts[1][1:])
        y = int(parts[2][1:])
        return x, y

    def make_loc(x, y):
        return f"loc-x{x}-y{y}"

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def manhattan(a, b):
        ax, ay = parse_loc(a)
        bx, by = parse_loc(b)
        return abs(ax - bx) + abs(ay - by)

    def neighbors(loc):
        x, y = parse_loc(loc)
        cands = []
        for dx, dy in product([0, 1, -1], [0, 1, -1]):
            if abs(dx) + abs(dy) != 1:
                continue
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny):
                nxt = make_loc(nx, ny)
                if nxt not in blocked:
                    cands.append(nxt)
        cands.sort(key=lambda n: (0 if n in visited else 1, manhattan(n, target), n))
        return cands

    if start == target:
        return [[]]

    base_dist = manhattan(start, target)
    max_steps = min(max(base_dist + 2, 1), width * height, 6)

    plans = []
    seen_paths = set()

    def path_to_actions(path):
        return [f"move {path[i]} {path[i+1]}" for i in range(len(path) - 1)]

    def dfs(path, used, steps_left):
        current = path[-1]
        if current == target:
            key = tuple(path)
            if key not in seen_paths:
                seen_paths.add(key)
                plans.append(path_to_actions(path))
            return
        if steps_left == 0:
            return

        # Prune if target cannot be reached within remaining steps.
        if manhattan(current, target) > steps_left:
            return

        for nxt in neighbors(current):
            if nxt in used:
                continue
            path.append(nxt)
            used.add(nxt)
            dfs(path, used, steps_left - 1)
            used.remove(nxt)
            path.pop()

    for steps in range(base_dist, max_steps + 1):
        dfs([start], {start}, steps)
        if plans:
            # Keep shortest-length candidate plans first, but allow multiple.
            shortest = min(len(p) for p in plans)
            filtered = [p for p in plans if len(p) == shortest]
            if filtered:
                return filtered

    return plans