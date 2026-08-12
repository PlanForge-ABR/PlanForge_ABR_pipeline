def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """
    Heuristic for the Depot domain.

    We approximate remaining work by counting unsatisfied goal facts, with
    higher weights for moving crates and loading/unloading trucks:

    - Missing `in(crate, truck)`       → cost 3
    - Missing `on(crate, support)`     → cost 2
    - Missing `at(obj, location)`      → cost 2
    - Any other missing goal predicate → cost 1

    This captures that satisfying cargo placement and vehicle locations is
    typically more involved than toggling simple flags like `clear` or
    `available`.
    """
    state_set = {(p.get("predicate"), tuple(p.get("args", []))) for p in state}

    total = 0
    for g in goal:
        key = (g.get("predicate"), tuple(g.get("args", [])))
        if key in state_set:
            continue

        pred = g.get("predicate")
        if pred == "in":
            total += 3
        elif pred == "on":
            total += 2
        elif pred == "at":
            total += 2
        else:
            total += 1

    return int(total)

