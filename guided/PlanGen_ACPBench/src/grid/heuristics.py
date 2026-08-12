from collections import deque, defaultdict

def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """Estimate cost by summing shortest distances for robots to reach 'at' goals.

    Conservative and fast: build graph from 'conn' and compute BFS distances from robot locations.
    """
    # build adjacency
    adj = defaultdict(list)
    robots = []
    for p in state:
        pred = p.get('predicate')
        args = p.get('args', [])
        if pred == 'conn' and len(args) >= 2:
            adj[args[0]].append(args[1])
        if pred == 'at-robot' and len(args) >= 1:
            # robots not named, just positions; treat as one robot per entry
            robots.append(args[0])

    def bfs(start):
        d = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    dist_maps = [bfs(r) for r in robots]

    total = 0
    # sum distances for 'at' goals (predicate 'at-robot' or 'at')
    for g in goal:
        if g.get('predicate') == 'at-robot' and len(g.get('args', [])) >= 1:
            tile = g['args'][0]
            best = min((dm.get(tile, float('inf')) for dm in dist_maps), default=float('inf'))
            total += (best if best != float('inf') else 1)
        else:
            total += 0
    return int(total)
