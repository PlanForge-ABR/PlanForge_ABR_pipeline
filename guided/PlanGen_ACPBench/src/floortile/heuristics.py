def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """Estimate remaining cost from `state` to `goal` for the floortile domain.

    Strategy (conservative, admissible-ish):
    - Build a directed adjacency graph from 'left','right','up','down' predicates.
    - Extract robot positions (`robot-at`), robot held colors (`robot-has`), available colors,
      and currently painted tiles.
    - For each unsatisfied goal predicate:
      - For `painted(tile,color)`: estimate = min over robots (distance(robot_pos, tile) + change_color_cost + 1 for paint)
        where change_color_cost is 0 if robot already has color, else 1 if color available.
      - For `robot-at(robot,tile)`: estimate = distance(robot_pos, tile).
      - For other predicates: small constant (1).
    - Sum these estimates.

    The heuristic is intentionally simple but captures travel + paint + change costs.
    """
    from collections import deque, defaultdict

    # helpers to search predicates
    def has_pred(state_list, pred, args):
        return {"predicate": pred, "args": args} in state_list

    # Build adjacency
    adj = defaultdict(list)
    robots = {}
    robot_has = {}
    avail_colors = set()
    painted = {}
    tiles = set()

    for p in state:
        pred = p.get('predicate')
        args = p.get('args', [])
        if pred in ('left', 'right', 'up', 'down') and len(args) >= 2:
            a, b = args[0], args[1]
            adj[a].append(b)
            # also register tiles
            tiles.add(a); tiles.add(b)
        elif pred == 'robot-at' and len(args) >= 2:
            robots[args[0]] = args[1]
            tiles.add(args[1])
        elif pred == 'robot-has' and len(args) >= 2:
            robot_has[args[0]] = args[1]
        elif pred == 'available-color' and len(args) >= 1:
            avail_colors.add(args[0])
        elif pred == 'painted' and len(args) >= 2:
            painted[args[0]] = args[1]
            tiles.add(args[0])
        else:
            # might include clear/other tile refs
            for a in args:
                if isinstance(a, str) and a.startswith('tile_'):
                    tiles.add(a)

    # BFS distance cache from a start tile
    def bfs_dist(start):
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # Precompute BFS distances from each robot position
    dist_cache = {}
    for r, pos in robots.items():
        dist_cache[r] = bfs_dist(pos)

    INF = 1e9
    total = 0.0

    # Build a quick lookup set for current state predicates for exact match checks
    state_set = { (p['predicate'], tuple(p.get('args', []))) for p in state }

    for g in goal:
        gp = g.get('predicate')
        gargs = g.get('args', [])
        key = (gp, tuple(gargs))
        if key in state_set:
            continue

        if gp == 'painted' and len(gargs) >= 2:
            tile, color = gargs[0], gargs[1]
            # If tile already painted with different color, we still need to paint
            best = INF
            # choose the robot that can achieve this cheapest
            for r, dmap in dist_cache.items():
                d = dmap.get(tile, None)
                if d is None:
                    continue
                change = 0 if robot_has.get(r) == color else (1 if color in avail_colors else 1)
                cost = d + change + 1  # travel + change + paint
                if cost < best:
                    best = cost
            if best == INF:
                # no robot can reach: pessimistic small constant
                best = 3
            total += best

        elif gp == 'robot-at' and len(gargs) >= 2:
            rname, tile = gargs[0], gargs[1]
            start = robots.get(rname)
            if start is None:
                total += 3
            else:
                # if we don't have precomputed dist for that robot, compute
                dmap = dist_cache.get(rname) or bfs_dist(start)
                dist_cache[rname] = dmap
                d = dmap.get(tile, None)
                total += (d if d is not None else 3)

        else:
            # generic unsatisfied predicate: small cost
            total += 1

    return int(total)
