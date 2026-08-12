import re

def plan_func(data, constraints):
    def parse_cell(cell):
        m = re.fullmatch(r"loc-x(\d+)-y(\d+)", str(cell).strip())
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def parse_action(step):
        if not isinstance(step, str):
            return None
        parts = step.strip().split()
        if len(parts) != 3 or parts[0] != "move":
            return None
        src, dst = parts[1], parts[2]
        if parse_cell(src) is None or parse_cell(dst) is None:
            return None
        return src, dst

    def is_neighbor(a, b):
        pa, pb = parse_cell(a), parse_cell(b)
        if pa is None or pb is None:
            return False
        return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) == 1

    def parse_goal_cell(state_query):
        if not state_query:
            return None
        m = re.search(r"(loc-x\d+-y\d+)", str(state_query))
        return m.group(1) if m else None

    # Infer the set of available cells from all candidate plans.
    available_cells = set()
    for plan in data or []:
        if not isinstance(plan, list):
            continue
        for step in plan:
            parsed = parse_action(step)
            if parsed:
                available_cells.update(parsed)

    goal_cell = parse_goal_cell(constraints.get("state_query"))

    for plan in data or []:
        if not isinstance(plan, list) or not plan:
            continue

        visited = set()
        current = None
        valid = True

        for i, step in enumerate(plan):
            parsed = parse_action(step)
            if not parsed:
                valid = False
                break

            src, dst = parsed

            if not is_neighbor(src, dst):
                valid = False
                break

            if i == 0:
                current = src
                visited.add(src)
            elif src != current:
                valid = False
                break

            current = dst
            visited.add(dst)

        if not valid:
            continue

        if goal_cell is not None and current != goal_cell:
            continue

        if available_cells and not available_cells.issubset(visited):
            continue

        return plan

    return None