def plan_func(data, constraints):
    import re

    def parse_pos(pos):
        m = re.fullmatch(r"f(\d+)-(\d+)f", pos)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def is_neighbor(a, b):
        pa, pb = parse_pos(a), parse_pos(b)
        if pa is None or pb is None:
            return False
        return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) == 1

    def in_grid(pos):
        p = parse_pos(pos)
        return p is not None and 0 <= p[0] < 5 and 0 <= p[1] < 5

    def parse_locked_positions(text):
        if not text:
            return set()
        return set(re.findall(r"f\d+-\d+f", str(text)))

    def parse_key_shape(key_name):
        m = re.fullmatch(r"key([^ ]+)", key_name)
        return m.group(1) if m else key_name

    def lock_matches_key(lock_pos, key_name):
        # Generic best-effort matcher:
        # if lock position text contains the key shape token, treat as matching.
        # If no shape mapping is available from constraints, this remains conservative.
        shape = parse_key_shape(key_name)
        return shape in lock_pos

    # Infer known key locations from all candidate plans.
    known_key_locations = {}
    for plan in data:
        if not isinstance(plan, list):
            continue
        for step in plan:
            parts = str(step).split()
            if len(parts) == 3 and parts[0] == "pickup":
                known_key_locations[parts[1]] = parts[2]

    locked_positions = parse_locked_positions(constraints.get("open_positions", ""))

    for plan in data:
        if not isinstance(plan, list) or not plan:
            continue

        valid = True
        current_pos = None
        arm_empty = str(constraints.get("arm_initial_state", "")).strip().lower() == "empty"
        held_key = None
        picked_keys = set()

        for i, step in enumerate(plan):
            parts = str(step).split()
            if not parts:
                valid = False
                break

            action = parts[0]

            if action == "move":
                if len(parts) != 3:
                    valid = False
                    break
                src, dst = parts[1], parts[2]

                if not in_grid(src) or not in_grid(dst):
                    valid = False
                    break
                if not is_neighbor(src, dst):
                    valid = False
                    break

                if current_pos is None:
                    current_pos = src
                if current_pos != src:
                    valid = False
                    break

                if dst in locked_positions:
                    if held_key is None or not lock_matches_key(dst, held_key):
                        valid = False
                        break

                current_pos = dst

            elif action == "pickup":
                if len(parts) != 3:
                    valid = False
                    break
                key_name, loc = parts[1], parts[2]

                if not in_grid(loc):
                    valid = False
                    break
                if current_pos is None or current_pos != loc:
                    valid = False
                    break
                if not arm_empty or held_key is not None:
                    valid = False
                    break

                known_loc = known_key_locations.get(key_name)
                if known_loc is not None and known_loc != loc:
                    valid = False
                    break
                if key_name in picked_keys:
                    valid = False
                    break

                held_key = key_name
                arm_empty = False
                picked_keys.add(key_name)

            else:
                valid = False
                break

        if valid:
            return plan

    return None