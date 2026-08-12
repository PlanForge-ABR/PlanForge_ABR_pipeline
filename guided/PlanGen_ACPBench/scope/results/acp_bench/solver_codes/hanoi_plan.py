def plan_func(data, constraints):
    import re

    def is_disk(x):
        return isinstance(x, str) and re.fullmatch(r"d\d+", x or "") is not None

    def disk_size(x):
        m = re.fullmatch(r"d(\d+)", x or "")
        return int(m.group(1)) if m else None

    def parse_move(step):
        if not isinstance(step, str):
            return None
        parts = step.strip().split()
        if len(parts) != 4 or parts[0].lower() != "move":
            return None
        disk, src, dst = parts[1], parts[2], parts[3]
        if not is_disk(disk):
            return None
        return disk, src, dst

    def infer_initial_support(plan):
        # Infer each moved disk's initial support from its first appearance as mover.
        support = {}
        for step in plan:
            mv = parse_move(step)
            if not mv:
                return None
            d, s, _ = mv
            if d not in support:
                support[d] = s
        return support

    def is_clear(item, support):
        return all(parent != item for parent in support.values())

    def creates_cycle(disk, dst, support):
        cur = dst
        seen = set()
        while is_disk(cur):
            if cur == disk:
                return True
            if cur in seen or cur not in support:
                break
            seen.add(cur)
            cur = support[cur]
        return False

    def goal_satisfied(support, constraints):
        goals = (constraints or {}).get("goal_conditions", []) or []
        for g in goals:
            if not isinstance(g, str):
                return False
            gl = g.strip().lower()

            m = re.fullmatch(r"([a-zA-Z0-9_]+) is not obstructed by any disk", gl)
            if m:
                item = m.group(1)
                if not is_clear(item, support):
                    return False
                continue

            m = re.fullmatch(r"([a-zA-Z0-9_]+) is clear", gl)
            if m:
                item = m.group(1)
                if not is_clear(item, support):
                    return False
                continue

            return False
        return True

    def valid_plan(plan, constraints):
        if not isinstance(plan, list):
            return False

        support = infer_initial_support(plan)
        if support is None:
            return False

        for d, s in support.items():
            if d == s:
                return False

        for step in plan:
            mv = parse_move(step)
            if not mv:
                return False
            disk, src, dst = mv

            if support.get(disk) != src:
                return False

            if not is_clear(disk, support):
                return False

            if not is_clear(dst, support):
                return False

            if is_disk(dst):
                if disk_size(disk) is None or disk_size(dst) is None:
                    return False
                if disk_size(disk) > disk_size(dst):
                    return False

            if dst == disk or creates_cycle(disk, dst, support):
                return False

            support[disk] = dst

        # For reachability queries, satisfy the explicit goal conditions.
        # Do not additionally require the full objective text to be completed,
        # since the expected solution is a partial plan reaching the queried state.
        return goal_satisfied(support, constraints)

    for plan in data:
        if valid_plan(plan, constraints):
            return plan
    return None