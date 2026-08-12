def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """
    Heuristic for the Swap domain.

    Agents are assigned items via `assigned(person, item)` and actions swap
    items between agents. A single swap can fix up to two incorrect
    assignments.

    We therefore:
    - Build current and goal mappings person → item from `assigned` facts.
    - Count how many agents have a different item than in the goal.
    - Estimate the minimum number of swaps as ceil(misplaced / 2).

    Static `not-eq` constraints are ignored in the heuristic (they are
    enforced by the transition model, not the goal).
    """
    current = {}
    goal_assign = {}

    for p in state:
        if p.get("predicate") == "assigned":
            args = p.get("args", [])
            if len(args) >= 2:
                person, item = args[0], args[1]
                current[person] = item

    for g in goal:
        if g.get("predicate") == "assigned":
            args = g.get("args", [])
            if len(args) >= 2:
                person, item = args[0], args[1]
                goal_assign[person] = item

    misplaced = 0
    for person, goal_item in goal_assign.items():
        if current.get(person) != goal_item:
            misplaced += 1

    # Each swap can correct at most two mis-assignments.
    swaps = (misplaced + 1) // 2
    return int(swaps)

