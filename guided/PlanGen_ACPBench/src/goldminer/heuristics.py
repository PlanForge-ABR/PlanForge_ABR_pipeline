from collections import defaultdict, deque


def _build_adjacency_and_features(state):
    """Extract grid adjacency and key features from the Goldminer state."""
    adj = defaultdict(list)
    robot_pos = None
    laser_cells = set()
    gold_cells = set()

    for p in state:
        pred = p.get("predicate")
        args = p.get("args", [])

        if pred == "connected" and len(args) >= 2:
            a, b = args[0], args[1]
            # Treat edges as undirected for distance estimation
            adj[a].append(b)
            adj[b].append(a)
        elif pred == "robot-at" and len(args) >= 1:
            robot_pos = args[0]
        elif pred == "laser-at" and len(args) >= 1:
            laser_cells.add(args[0])
        elif pred == "gold-at" and len(args) >= 1:
            gold_cells.add(args[0])

    return adj, robot_pos, laser_cells, gold_cells


def _bfs_distances(adj, start):
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """
    Heuristic for the Goldminer (bomberman-style) domain.

    We use the underlying grid from `connected` predicates and estimate:

    - Distance for the robot to reach goal positions (`robot-at` goals).
    - Distance to reach a laser or gold cell for `holds-laser` / `holds-gold`.
    - A small unit cost for any other unsatisfied goal predicate.

    This leverages the spatial structure of the domain while remaining cheap.
    """
    state_set = {(p.get("predicate"), tuple(p.get("args", []))) for p in state}
    adj, robot_pos, laser_cells, gold_cells = _build_adjacency_and_features(state)

    dist = {}
    if robot_pos is not None:
        dist = _bfs_distances(adj, robot_pos)

    INF = 10**6
    total = 0

    for g in goal:
        pred = g.get("predicate")
        args = g.get("args", [])
        key = (pred, tuple(args))

        if key in state_set:
            continue

        if pred == "robot-at" and len(args) >= 1:
            target = args[0]
            d = dist.get(target, None) if dist else None
            total += d if d is not None else 3

        elif pred == "holds-laser":
            # Need to reach some laser cell, then pick it up.
            best = INF
            if dist and laser_cells:
                for cell in laser_cells:
                    if cell in dist:
                        best = min(best, dist[cell] + 1)  # move + pickup
            total += best if best != INF else 4

        elif pred == "holds-gold":
            # Need to reach some gold cell, then pick it up.
            best = INF
            if dist and gold_cells:
                for cell in gold_cells:
                    if cell in dist:
                        best = min(best, dist[cell] + 1)
            total += best if best != INF else 4

        else:
            # Generic unsatisfied goal: assign small cost.
            total += 1

    return int(total)

