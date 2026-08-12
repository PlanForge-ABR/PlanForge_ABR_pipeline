def heuristic(state, goal):
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    if isinstance(goal, dict) and "state" in goal:
        goal = goal["state"]

    state = state or []
    goal = goal or []

    facts = {(p.get("predicate"), tuple(p.get("args", []))) for p in state}

    # Fast-path: already a goal state.
    if all((g.get("predicate"), tuple(g.get("args", []))) in facts for g in goal):
        return 0

    frog_pos = {}
    occupied = set()
    goal_pos = {}
    goal_empty = set()

    next_right = {}
    next_left = {}

    for fact in state:
        pred = fact.get("predicate")
        args = fact.get("args", [])
        if pred == "at" and len(args) == 2:
            frog, pos = args
            frog_pos[frog] = pos
            occupied.add(pos)
        elif pred == "next" and len(args) == 2:
            a, b = args
            next_right[a] = b
            next_left[b] = a

    # If 'next' isn't present in the state, try to recover it from the goal.
    if not next_right:
        for fact in goal:
            if fact.get("predicate") == "next":
                args = fact.get("args", [])
                if len(args) == 2:
                    a, b = args
                    next_right[a] = b
                    next_left[b] = a

    for fact in goal:
        pred = fact.get("predicate")
        args = fact.get("args", [])
        if pred == "at" and len(args) == 2:
            frog, pos = args
            goal_pos[frog] = pos
        elif pred == "empty" and len(args) == 1:
            goal_empty.add(args[0])

    def build_position_index():
        if not next_right:
            return None
        nodes = set(next_right.keys()) | set(next_right.values())
        # Leftmost position has no predecessor.
        start = None
        for n in nodes:
            if n not in next_left:
                start = n
                break
        if start is None:
            start = next(iter(nodes))
        order = []
        cur = start
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            order.append(cur)
            cur = next_right.get(cur)
        return {p: i for i, p in enumerate(order)}

    pos_index = build_position_index()

    total_distance = 0
    if pos_index:
        for frog, gpos in goal_pos.items():
            cpos = frog_pos.get(frog)
            if cpos in pos_index and gpos in pos_index:
                total_distance += abs(pos_index[cpos] - pos_index[gpos])

    # One move can reduce the total distance by at most 2 (slide=1, jump=2),
    # so ceil(total_distance / 2) is admissible (and typically consistent).
    h_distance = (total_distance + 1) // 2

    # If certain positions must be empty, at least one move is needed per
    # currently-occupied "must-be-empty" position (a single move can fix at most one).
    occupied_must_be_empty = sum(1 for p in goal_empty if p in occupied)

    h_value = max(h_distance, occupied_must_be_empty)
    if h_value == 0:
        # Fallback when we couldn't compute distances but goal is not yet met.
        h_value = 1
    return int(h_value)

