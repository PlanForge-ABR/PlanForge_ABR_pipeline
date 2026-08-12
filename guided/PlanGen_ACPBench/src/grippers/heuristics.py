def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """Simple heuristic: count number of goal 'holding' predicates not met.
    This gives a small admissible estimate for grippers tasks.
    """
    goal_set = {(p['predicate'], tuple(p.get('args', []))) for p in goal}
    state_set = {(p['predicate'], tuple(p.get('args', []))) for p in state}
    missing = [g for g in goal_set if g not in state_set]
    return len(missing)
