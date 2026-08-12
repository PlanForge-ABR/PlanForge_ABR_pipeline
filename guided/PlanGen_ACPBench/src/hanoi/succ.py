def successor(state):
    """
    Generate all legal (action, next_state) pairs for the Towers of Hanoi domain.

    State is a list of dicts like: {"predicate": "on", "args": ["d1", "d6"]}.
    Predicates used: on(x,y), clear(x), smaller(a,b) meaning b can be placed on a.
    """
    facts = {(p["predicate"], tuple(p["args"])) for p in state}

    clear_set = {args[0] for pred, args in facts if pred == "clear" and len(args) == 1}
    on_of = {args[0]: args[1] for pred, args in facts if pred == "on" and len(args) == 2}
    smaller_set = {args for pred, args in facts if pred == "smaller" and len(args) == 2}

    results = []

    # Movable items are those that are clear and are currently on something.
    for moving in clear_set:
        from_obj = on_of.get(moving)
        if from_obj is None:
            continue

        for to_obj in clear_set:
            if to_obj == moving or to_obj == from_obj:
                continue
            # Check smaller constraint
            is_smaller = False
            if (to_obj, moving) in smaller_set:
                is_smaller = True
            else:
                # Fallback: parse names
                # smaller(dest, moving) means moving can be placed on dest
                # If dest is a peg, it's always allowed (assuming standard Hanoi)
                if to_obj.lower().startswith('peg'):
                    is_smaller = True
                elif to_obj.lower().startswith('d') and moving.lower().startswith('d'):
                    try:
                        to_idx = int(to_obj[1:])
                        moving_idx = int(moving[1:])
                        # In standard hanoi, dLarge can hold dSmall
                        if to_idx > moving_idx:
                            is_smaller = True
                    except ValueError:
                        pass
            
            if not is_smaller:
                continue

            next_facts = set(facts)
            next_facts.discard(("on", (moving, from_obj)))
            next_facts.discard(("clear", (to_obj,)))
            next_facts.add(("on", (moving, to_obj)))
            next_facts.add(("clear", (from_obj,)))

            next_state = [
                {"predicate": pred, "args": list(args)}
                for pred, args in sorted(next_facts, key=lambda x: (x[0], x[1]))
            ]
            results.append((f"move {moving} {from_obj} {to_obj}", next_state))

    return results

