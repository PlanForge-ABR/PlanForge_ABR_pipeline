def is_goal(state, goal):
    """Return True if all goal predicates are present in state.

    Both `state` and `goal` are lists of dicts with 'predicate' and 'args'.
    """
    sset = set((p['predicate'], tuple(p['args'])) for p in state)
    for g in goal:
        if (g['predicate'], tuple(g['args'])) not in sset:
            return False
    return True
