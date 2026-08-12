def successor(state):
    """
    Generate successor states for the frogs-jumping domain.

    Input `state` is a list of dict predicates, including:
      - {"predicate": "at", "args": [frog, position]}
      - {"predicate": "empty", "args": [position]}
      - {"predicate": "next", "args": [position, position]}  (directed left->right)

    Output is a list of (action_string, next_state) pairs.
    """

    at_pos_by_frog = {}
    frog_by_pos = {}
    empty_positions = set()
    next_right = {}
    next_left = {}

    for fact in state or []:
        pred = fact.get("predicate")
        args = fact.get("args", [])
        if pred == "at" and len(args) == 2:
            frog, pos = args
            at_pos_by_frog[frog] = pos
            frog_by_pos[pos] = frog
        elif pred == "empty" and len(args) == 1:
            empty_positions.add(args[0])
        elif pred == "next" and len(args) == 2:
            a, b = args
            next_right[a] = b
            next_left[b] = a

    def make_state(moving_frog, from_pos, to_pos):
        next_state = []
        for fact in state:
            pred = fact.get("predicate")
            args = fact.get("args", [])
            if pred == "at" and len(args) == 2 and args[0] == moving_frog:
                next_state.append({"predicate": "at", "args": [moving_frog, to_pos]})
            elif pred == "empty" and len(args) == 1 and args[0] == to_pos:
                continue
            else:
                next_state.append({"predicate": pred, "args": list(args)})
        next_state.append({"predicate": "empty", "args": [from_pos]})
        return next_state

    successors = []

    for frog, from_pos in at_pos_by_frog.items():
        is_left_frog = frog.startswith("l")
        is_right_frog = frog.startswith("r")
        if not (is_left_frog or is_right_frog):
            continue

        if is_left_frog:
            mid = next_right.get(from_pos)
            if mid and mid in empty_positions:
                action = f"slide-right {frog} {from_pos} {mid}"
                successors.append((action, make_state(frog, from_pos, mid)))

            if mid and mid in frog_by_pos:
                to_pos = next_right.get(mid)
                if to_pos and to_pos in empty_positions:
                    jumped_frog = frog_by_pos[mid]
                    if jumped_frog.startswith("r"):
                        action = f"jump-right {frog} {from_pos} {mid} {to_pos} {jumped_frog}"
                        successors.append((action, make_state(frog, from_pos, to_pos)))

        if is_right_frog:
            mid = next_left.get(from_pos)
            if mid and mid in empty_positions:
                action = f"slide-left {frog} {from_pos} {mid}"
                successors.append((action, make_state(frog, from_pos, mid)))

            if mid and mid in frog_by_pos:
                to_pos = next_left.get(mid)
                if to_pos and to_pos in empty_positions:
                    jumped_frog = frog_by_pos[mid]
                    if jumped_frog.startswith("l"):
                        action = f"jump-left {frog} {from_pos} {mid} {to_pos} {jumped_frog}"
                        successors.append((action, make_state(frog, from_pos, to_pos)))

    return successors
