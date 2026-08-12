def plan_func(data, constraints):
    def pos_num(pos):
        if not isinstance(pos, str):
            return None
        digits = "".join(ch for ch in pos if ch.isdigit())
        return int(digits) if digits else None

    def parse_action(step):
        if isinstance(step, str):
            tokens = step.strip().split()
        elif isinstance(step, (list, tuple)):
            tokens = list(step)
        else:
            return None

        if len(tokens) < 4:
            return None

        action = tokens[0]
        piece = tokens[1]
        src = tokens[2]
        dst = tokens[3]
        return action, piece, src, dst

    def is_valid_step(action, piece, src, dst, pos_to_piece):
        # Piece must be at source
        if pos_to_piece.get(src) != piece:
            return False

        # Destination must be empty
        if dst in pos_to_piece:
            return False

        s_num = pos_num(src)
        d_num = pos_num(dst)

        # If positions are numeric, validate directional/action semantics when possible
        if s_num is not None and d_num is not None:
            diff = d_num - s_num
            action_l = action.lower()

            if "left" in action_l and not (diff < 0):
                return False
            if "right" in action_l and not (diff > 0):
                return False

            if "slide" in action_l:
                if abs(diff) != 1:
                    return False
            elif "jump" in action_l or "hop" in action_l:
                if abs(diff) != 2:
                    return False
                mid = "p{}".format((s_num + d_num) // 2)
                if mid not in pos_to_piece:
                    return False

        return True

    current_state = constraints.get("current_state", {}) or {}
    initial_pos_to_piece = {pos: piece for piece, pos in current_state.items()}

    for plan in data or []:
        if not isinstance(plan, list):
            continue

        pos_to_piece = dict(initial_pos_to_piece)
        valid = True

        for step in plan:
            parsed = parse_action(step)
            if parsed is None:
                valid = False
                break

            action, piece, src, dst = parsed

            if not is_valid_step(action, piece, src, dst, pos_to_piece):
                valid = False
                break

            # Apply move
            del pos_to_piece[src]
            pos_to_piece[dst] = piece

        if valid:
            return plan

    return None