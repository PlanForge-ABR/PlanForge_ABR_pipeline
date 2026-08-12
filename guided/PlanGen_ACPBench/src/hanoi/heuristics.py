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

    on_of = {}
    supported_by = {}
    for p in state:
        if p.get("predicate") == "on":
            args = p.get("args", [])
            if len(args) == 2:
                top, bottom = args
                on_of[top] = bottom
                # In standard Hanoi encodings, each object supports at most one item.
                supported_by[bottom] = top

    def count_above(obj):
        cnt = 0
        cur = obj
        seen = set()
        while cur in supported_by and cur not in seen:
            seen.add(cur)
            cnt += 1
            cur = supported_by[cur]
        return cnt

    best = 0
    for g in goal:
        pred = g.get("predicate")
        args = g.get("args", [])
        g_fact = (pred, tuple(args))

        if g_fact in facts:
            continue

        if pred == "on" and len(args) == 2:
            disk, target = args
            lb = 1 + count_above(disk) + count_above(target)
            best = max(best, lb)
            continue

        if pred == "clear" and len(args) == 1:
            obj = args[0]
            lb = count_above(obj)
            best = max(best, lb if lb > 0 else 1)
            continue

        # Unknown predicate types: if missing, we at least need 1 move.
        best = max(best, 1)

    return int(best)

