def combinations_func(data):
    from itertools import product

    target_assignments = data.get("target_assignments", [])

    def collect_assignments(container):
        facts = []
        if isinstance(container, dict):
            for key in (
                "initial_assignments",
                "assignments",
                "state",
                "facts",
                "init",
                "initial_state",
                "current_state",
            ):
                val = container.get(key)
                if isinstance(val, list):
                    facts.extend(val)
        return facts

    facts = collect_assignments(data)

    def get_num(name):
        if not isinstance(name, str):
            return None
        digits = ""
        for ch in reversed(name):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        return int(digits) if digits else None

    def build_at_map(assignments):
        at_map = {}
        empty_map = {}
        for a in assignments:
            if not isinstance(a, dict):
                continue
            rel = a.get("relation")
            if rel == "at":
                at_map[a.get("object")] = a.get("value")
            elif rel == "empty":
                empty_map[a.get("object")] = a.get("value")
        return at_map, empty_map

    at_map, empty_map = build_at_map(facts)

    target_at = {}
    target_empty = set()
    for a in target_assignments:
        if not isinstance(a, dict):
            continue
        if a.get("relation") == "at":
            target_at[a.get("object")] = a.get("value")
        elif a.get("relation") == "empty" and a.get("value") is True:
            target_empty.add(a.get("object"))

    # Optional adjacency from data
    adjacency = {}
    for key in ("adjacency", "neighbors", "graph"):
        if isinstance(data.get(key), dict):
            adjacency = data[key]
            break

    width = data.get("grid_width") or data.get("width") or data.get("cols")

    def infer_direction(src, dst):
        if src is None or dst is None:
            return None
        if adjacency:
            if dst in adjacency.get(src, []):
                s, d = get_num(src), get_num(dst)
                if s is not None and d is not None:
                    if d == s - 1:
                        return "slide-left"
                    if d == s + 1:
                        return "slide-right"
                    if width:
                        if d == s - width:
                            return "slide-up"
                        if d == s + width:
                            return "slide-down"
                return "slide"
        s, d = get_num(src), get_num(dst)
        if s is None or d is None:
            return None
        if d == s - 1:
            return "slide-left"
        if d == s + 1:
            return "slide-right"
        if width:
            if d == s - width:
                return "slide-up"
            if d == s + width:
                return "slide-down"
        return None

    def neighbors(pos):
        if adjacency and pos in adjacency:
            return list(adjacency[pos])
        n = get_num(pos)
        if n is None:
            return []
        out = []
        for delta in (-1, 1):
            out.append(f"p{n + delta}")
        if width:
            out.append(f"p{n - width}")
            out.append(f"p{n + width}")
        return out

    plans = []

    # Primary pattern: move target object from its current place into target place,
    # making its source place empty.
    for obj, dst in target_at.items():
        src = at_map.get(obj)
        candidate_sources = set(target_empty)
        if src:
            candidate_sources.add(src)

        # 1-step plans
        for source in candidate_sources:
            if source and dst:
                action = infer_direction(source, dst)
                if action:
                    plans.append([f"{action} {obj} {source} {dst}"])

        # 2-step plans via one intermediate
        if src and dst:
            mids = set(neighbors(src)) | set(neighbors(dst))
            for mid in mids:
                a1 = infer_direction(src, mid)
                a2 = infer_direction(mid, dst)
                if a1 and a2 and mid not in (src, dst):
                    plans.append([
                        f"{a1} {obj} {src} {mid}",
                        f"{a2} {obj} {mid} {dst}",
                    ])

    # Fallback: if no initial location is known, synthesize from target empty place.
    if not plans:
        for obj, dst in target_at.items():
            for src in target_empty:
                action = infer_direction(src, dst)
                if action:
                    plans.append([f"{action} {obj} {src} {dst}"])

    # Deduplicate while preserving order
    seen = set()
    unique_plans = []
    for plan in plans:
        key = tuple(plan)
        if key not in seen:
            seen.add(key)
            unique_plans.append(plan)

    return unique_plans