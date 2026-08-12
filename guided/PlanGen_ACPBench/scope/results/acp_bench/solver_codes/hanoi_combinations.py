def combinations_func(data):
    from itertools import product

    def disk_num(x):
        return int(x[1:]) if isinstance(x, str) and x.startswith("d") and x[1:].isdigit() else None

    def is_disk(x):
        return disk_num(x) is not None

    def is_peg(x):
        return isinstance(x, str) and x.startswith("peg")

    def parse_state(state):
        support_of = {}
        occupied_supports = set()
        for top, bottom in state.get("on_relations", []):
            support_of[top] = bottom
            occupied_supports.add(bottom)

        all_disks = [f"d{i}" for i in range(1, data.get("num_disks", 0) + 1)]
        pegs = [f"peg{i}" for i in range(1, data.get("num_pegs", 0) + 1)]

        clear = set(state.get("clear_items", []))
        # If clear items are incomplete, infer them.
        for d in all_disks:
            if d in support_of and d not in occupied_supports:
                clear.add(d)
        for p in pegs:
            if p not in occupied_supports:
                clear.add(p)

        return {
            "support_of": support_of,
            "clear": clear,
            "disks": all_disks,
            "pegs": pegs,
        }

    def legal_moves(st):
        moves = []
        clear = st["clear"]
        support_of = st["support_of"]

        clear_disks = [x for x in clear if is_disk(x)]
        clear_pegs = [x for x in clear if is_peg(x)]

        movable_disks = [d for d in clear_disks if d in support_of]

        for d in movable_disks:
            src = support_of[d]
            dnum = disk_num(d)

            for dst in clear_pegs + clear_disks:
                if dst == d or dst == src:
                    continue
                if is_disk(dst):
                    dst_num = disk_num(dst)
                    if dst_num is None or dnum is None or dnum > dst_num:
                        continue
                moves.append((d, src, dst))
        return moves

    def apply_move(st, move):
        d, src, dst = move
        new_support = dict(st["support_of"])
        new_clear = set(st["clear"])

        new_support[d] = dst

        new_clear.add(src)
        if dst in new_clear:
            new_clear.remove(dst)
        new_clear.add(d)

        return {
            "support_of": new_support,
            "clear": new_clear,
            "disks": st["disks"],
            "pegs": st["pegs"],
        }

    def goal_satisfied(st, goal):
        goal_clear = set(goal.get("clear_items", []))
        return goal_clear.issubset(st["clear"])

    def move_to_str(move):
        return f"move {move[0]} {move[1]} {move[2]}"

    init = parse_state(data.get("initial_state", {}))
    goal = data.get("goal_state", {})

    max_depth = 4
    candidates = []

    # BFS-style expansion up to small depth.
    frontier = [(init, [])]
    seen = set()

    def state_key(st):
        return (
            tuple(sorted(st["support_of"].items())),
            tuple(sorted(st["clear"])),
        )

    for depth in range(1, max_depth + 1):
        new_frontier = []
        for st, plan in frontier:
            moves = legal_moves(st)
            for mv in moves:
                st2 = apply_move(st, mv)
                plan2 = plan + [move_to_str(mv)]
                key = (state_key(st2), tuple(plan2))
                if key in seen:
                    continue
                seen.add(key)

                if goal_satisfied(st2, goal):
                    candidates.append(plan2)

                new_frontier.append((st2, plan2))
        frontier = new_frontier

    # If nothing satisfies the goal, still return short legal candidate plans.
    if not candidates:
        st = init
        first_moves = legal_moves(st)
        for mv1 in first_moves:
            plan1 = [move_to_str(mv1)]
            candidates.append(plan1)
            st1 = apply_move(st, mv1)
            for mv2 in legal_moves(st1):
                candidates.append(plan1 + [move_to_str(mv2)])
                if len(candidates) >= 20:
                    break
            if len(candidates) >= 20:
                break

    # Deduplicate while preserving order.
    unique = []
    seen_plans = set()
    for p in candidates:
        t = tuple(p)
        if t not in seen_plans:
            seen_plans.add(t)
            unique.append(p)

    return unique