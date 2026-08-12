def plan_func(data, constraints):
    def parse_pos(pos):
        if not isinstance(pos, str):
            return None
        pos = pos.strip()
        if not (pos.startswith("f") and pos.endswith("f")):
            return None
        core = pos[1:-1]
        if "-" not in core:
            return None
        a, b = core.split("-", 1)
        if not (a.isdigit() and b.isdigit()):
            return None
        return (int(a), int(b))

    def parse_grid_size(s):
        if not isinstance(s, str) or "x" not in s:
            return None
        a, b = s.lower().split("x", 1)
        if not (a.isdigit() and b.isdigit()):
            return None
        return int(a), int(b)

    def in_bounds(p, grid):
        return p is not None and 0 <= p[0] < grid[0] and 0 <= p[1] < grid[1]

    def is_neighbor(a, b):
        return a is not None and b is not None and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def blocked(pos, hard_rocks, soft_rocks):
        return pos in hard_rocks or pos in soft_rocks

    def normalize_action(action):
        if not isinstance(action, str):
            return None, []
        parts = action.strip().split()
        if not parts:
            return None, []
        return parts[0].lower(), parts[1:]

    grid = parse_grid_size(constraints.get("grid_size", ""))
    if grid is None:
        return None

    start_pos = parse_pos(constraints.get("robot_initial_position"))
    gold_pos = parse_pos(constraints.get("gold_location"))
    laser_pos = parse_pos(constraints.get("laser_location"))
    bomb_pos = parse_pos(constraints.get("bomb_supply_location"))

    if not in_bounds(start_pos, grid):
        return None

    hard_rocks_init = {parse_pos(p) for p in constraints.get("hard_rock_locations", [])}
    soft_rocks_init = {parse_pos(p) for p in constraints.get("soft_rock_locations", [])}
    hard_rocks_init.discard(None)
    soft_rocks_init.discard(None)

    initial_holding = constraints.get("robot_initial_holding", "empty")
    laser_qty = int(constraints.get("laser_quantity", 0) or 0)

    for plan in data:
        if not isinstance(plan, list):
            continue

        robot_pos = start_pos
        holding = initial_holding
        gold_picked = False
        laser_available = laser_qty
        bomb_available = bomb_pos is not None
        hard_rocks = set(hard_rocks_init)
        soft_rocks = set(soft_rocks_init)

        valid = True

        for step in plan:
            op, args = normalize_action(step)
            if op is None:
                valid = False
                break

            if op == "move":
                if len(args) != 2:
                    valid = False
                    break
                src = parse_pos(args[0])
                dst = parse_pos(args[1])
                if src != robot_pos:
                    valid = False
                    break
                if not in_bounds(dst, grid) or not is_neighbor(src, dst):
                    valid = False
                    break
                if blocked(dst, hard_rocks, soft_rocks):
                    valid = False
                    break
                robot_pos = dst

            elif op == "pick-gold":
                if len(args) != 1:
                    valid = False
                    break
                loc = parse_pos(args[0])
                if loc != robot_pos or loc != gold_pos or gold_picked:
                    valid = False
                    break
                if holding != "empty":
                    valid = False
                    break
                holding = "gold"
                gold_picked = True

            elif op == "pick-laser":
                if len(args) != 1:
                    valid = False
                    break
                loc = parse_pos(args[0])
                if loc != robot_pos or loc != laser_pos:
                    valid = False
                    break
                if holding != "empty":
                    valid = False
                    break
                if laser_available <= 0:
                    valid = False
                    break
                holding = "laser"
                laser_available -= 1

            elif op == "pick-bomb":
                if len(args) != 1:
                    valid = False
                    break
                loc = parse_pos(args[0])
                if loc != robot_pos or loc != bomb_pos:
                    valid = False
                    break
                if holding != "empty":
                    valid = False
                    break
                if not bomb_available:
                    valid = False
                    break
                holding = "bomb"
                bomb_available = False

            elif op in ("fire-laser", "use-laser"):
                # Supported forms:
                #   fire-laser target
                #   fire-laser source target
                if holding != "laser":
                    valid = False
                    break
                if len(args) == 1:
                    src = robot_pos
                    tgt = parse_pos(args[0])
                elif len(args) == 2:
                    src = parse_pos(args[0])
                    tgt = parse_pos(args[1])
                    if src != robot_pos:
                        valid = False
                        break
                else:
                    valid = False
                    break
                if not in_bounds(tgt, grid) or not is_neighbor(src, tgt):
                    valid = False
                    break
                if tgt in hard_rocks:
                    hard_rocks.remove(tgt)
                elif tgt in soft_rocks:
                    soft_rocks.remove(tgt)
                else:
                    valid = False
                    break

            elif op in ("detonate-bomb", "use-bomb"):
                # Supported forms:
                #   detonate-bomb target
                #   detonate-bomb source target
                if holding != "bomb":
                    valid = False
                    break
                if len(args) == 1:
                    src = robot_pos
                    tgt = parse_pos(args[0])
                elif len(args) == 2:
                    src = parse_pos(args[0])
                    tgt = parse_pos(args[1])
                    if src != robot_pos:
                        valid = False
                        break
                else:
                    valid = False
                    break
                if not in_bounds(tgt, grid) or not is_neighbor(src, tgt):
                    valid = False
                    break
                if tgt not in soft_rocks:
                    valid = False
                    break
                soft_rocks.remove(tgt)
                holding = "empty"

            elif op in ("drop", "place-bomb", "put-down"):
                # Constraints say bomb cannot be placed back; no generic dropping allowed.
                valid = False
                break

            else:
                valid = False
                break

        if valid:
            return plan

    return None