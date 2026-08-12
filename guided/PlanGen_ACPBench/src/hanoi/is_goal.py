def is_goal(current_state, goal_state):
    """
    Return True iff every predicate in goal_state appears in current_state.

    States are lists of dicts like: {"predicate": "on", "args": ["d1", "peg1"]}.
    """
    current_facts = {(p["predicate"], tuple(p["args"])) for p in current_state}
    for p in goal_state:
        if (p["predicate"], tuple(p["args"])) not in current_facts:
            return False
    return True

