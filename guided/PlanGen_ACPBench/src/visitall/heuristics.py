def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    sset = set((p['predicate'], tuple(p['args'])) for p in state)
    missing = 0
    for g in goal:
        if (g['predicate'], tuple(g['args'])) not in sset:
            missing += 1
    return missing
