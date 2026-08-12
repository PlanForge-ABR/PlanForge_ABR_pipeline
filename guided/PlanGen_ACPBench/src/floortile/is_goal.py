def is_goal(given_state, goal_state):
    """Return True if all predicates in goal_state are present in given_state.
    The given_state may contain extra predicates.
    """
    for g in goal_state:
        if g not in given_state:
            return False
    return True
