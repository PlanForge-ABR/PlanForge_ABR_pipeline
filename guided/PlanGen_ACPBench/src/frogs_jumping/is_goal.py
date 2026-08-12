def is_goal(current_state, goal_state):
    """
    Return True iff every predicate in `goal_state` appears in `current_state`.

    States are lists of dicts like: {"predicate": "at", "args": ["l1", "p1"]}.
    Goal states in this domain are partial (typically only `at`/`empty` facts).
    """

    def to_fact(item):
        predicate = item.get("predicate")
        args = item.get("args", [])
        return (predicate, tuple(args))

    current_facts = {to_fact(p) for p in (current_state or [])}
    for p in goal_state or []:
        if to_fact(p) not in current_facts:
            return False
    return True

