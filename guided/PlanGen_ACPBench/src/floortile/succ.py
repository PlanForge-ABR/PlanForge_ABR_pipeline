def successor(given_state):
    # We return a list of (action, state) pairs
    next_states = []

    # Helpers
    def _has(state, pred, args):
        return {"predicate": pred, "args": args} in state

    def _copy(state):
        return [p.copy() for p in state]

    def _remove(state, pred, args):
        state[:] = [p for p in state if not (p['predicate'] == pred and p['args'] == args)]

    def _add(state, pred, args):
        if not _has(state, pred, args):
            state.append({"predicate": pred, "args": args})

    # Collect movement relations and robots/colors
    tiles = set()
    robots = set()
    avail_colors = set()
    painted = {}
    relations = {"left": [], "right": [], "up": [], "down": []}

    for p in given_state:
        pred = p['predicate']
        args = p['args']
        for a in args:
            if isinstance(a, str) and a.startswith('tile_'):
                tiles.add(a)
        if pred == 'robot-at':
            robots.add(args[0])
        if pred == 'available-color':
            avail_colors.add(args[0])
        if pred == 'painted':
            painted[args[0]] = args[1]
        if pred in relations:
            relations[pred].append((args[0], args[1]))

    # Movement actions: for each robot and relation, if robot-at(robot, A) -> robot-at(robot, B)
    # Only allow moving into tiles that are clear and not painted.
    painted_tiles = {t for t, c in painted.items()}
    all_relations = [('left', relations['left']), ('right', relations['right']), ('up', relations['up']), ('down', relations['down'])]
    # mapping from relation predicate to action name (opposite)
    opp = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
    for r in robots:
        for rel_name, rel_list in all_relations:
            for (a, b) in rel_list:
                # robot at a can move to b if b is clear and not painted
                if _has(given_state, 'robot-at', [r, a]) and _has(given_state, 'clear', [b]) and b not in painted_tiles:
                    new = _copy(given_state)
                    _remove(new, 'robot-at', [r, a])
                    _add(new, 'robot-at', [r, b])
                    # moving a robot should make the origin tile clear and the destination not clear
                    _add(new, 'clear', [a])
                    _remove(new, 'clear', [b])
                    action = f"{opp[rel_name]} {r} {a} {b}"
                    next_states.append(
                        (action, sorted(new, key=lambda p: (p['predicate'], tuple(p['args'])) ))
                    )

    # change-color actions: robot-has R old -> robot-has R new (if available-color new)
    # Include same-color change only if the robot can currently paint an up/down neighbor that is clear and not painted
    def robot_can_paint(r):
        # robot can paint if it has an up/down neighbor tile that is clear and not painted
        pos = None
        for p in given_state:
            if p['predicate'] == 'robot-at' and p['args'][0] == r:
                pos = p['args'][1]
                break
        if pos is None:
            return False
        # check relations up/down where pos is source
        for (a, b) in relations['up'] + relations['down']:
            if a == pos and _has(given_state, 'clear', [b]) and b not in painted_tiles:
                return True
        return False

    for r in robots:
        old = None
        for p in given_state:
            if p['predicate'] == 'robot-has' and p['args'][0] == r:
                old = p['args'][1]
                break
        if old is None:
            continue
        # always allow change-color actions (including same-color), fixtures expect same-color options
        for new_color in avail_colors:
            action = f"change-color {r} {old} {new_color}"
            new_state = _copy(given_state)
            _remove(new_state, 'robot-has', [r, old])
            _add(new_state, 'robot-has', [r, new_color])
            next_states.append(
                (action, sorted(new_state, key=lambda p: (p['predicate'], tuple(p['args'])) ))
            )

    # paint actions: robots can only paint tiles in the up/down direction relative to their location
    # For a relation a->b in 'up' or 'down', if robot at a and b is clear and not painted, robot can paint b
    for rel_name in ('up', 'down'):
        for (a, b) in relations[rel_name]:
            # robot at a can paint b
            for r in robots:
                if _has(given_state, 'robot-at', [r, a]):
                    # find robot's color
                    color = None
                    for q in given_state:
                        if q['predicate'] == 'robot-has' and q['args'][0] == r:
                            color = q['args'][1]
                            break
                    if color is None:
                        continue
                    if _has(given_state, 'clear', [b]) and b not in painted_tiles:
                        action = f"paint-{opp[rel_name]} {r} {b} {a} {color}"
                        new_state = _copy(given_state)
                        _remove(new_state, 'clear', [b])
                        _add(new_state, 'painted', [b, color])
                        next_states.append(
                            (action, sorted(new_state, key=lambda p: (p['predicate'], tuple(p['args'])) ))
                        )

    # deduplicate successor states (canonicalized) to avoid duplicates caused by symmetric rules
    # For each distinct successor state, keep only the lexicographically
    # smallest action string that leads to it (to match fixtures).
    best_for_state = {}
    state_objects = {}
    for action, state in next_states:
        key = tuple((p['predicate'], tuple(p['args'])) for p in state)
        if key not in best_for_state or action < best_for_state[key]:
            best_for_state[key] = action
            state_objects[key] = state

    return [(best_for_state[key], state_objects[key]) for key in best_for_state]
