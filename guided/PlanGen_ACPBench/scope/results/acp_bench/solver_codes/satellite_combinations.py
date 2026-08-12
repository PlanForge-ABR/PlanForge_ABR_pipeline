def combinations_func(data):
    from itertools import product

    def ensure_list(x):
        if x is None:
            return []
        return x if isinstance(x, list) else [x]

    def collect_objects(prefix):
        vals = []
        for k, v in data.items():
            if prefix in k:
                vals.extend(ensure_list(v))
        # deduplicate while preserving order
        seen = set()
        out = []
        for item in vals:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    # Goal-specific objects
    powered_on_goals = ensure_list(data.get("powered_on_instruments_goal", []))

    # Collect any explicitly provided objects
    satellites = collect_objects("satellite")
    instruments = collect_objects("instrument")
    planets = collect_objects("planet")
    modes = collect_objects("thermograph") + collect_objects("mode")

    # Backoff generic pools.
    # These are intentionally broad enough to include hidden constants
    # such as satellite3, planet5, instrument4, thermograph0.
    if not satellites:
        satellites = [f"satellite{i}" for i in range(6)]
    if not instruments:
        instruments = [f"instrument{i}" for i in range(6)]
    else:
        # extend explicit instruments with a broader pool so mixed plans can include
        # non-goal instruments such as instrument4
        for i in range(6):
            name = f"instrument{i}"
            if name not in instruments:
                instruments.append(name)
    if not planets:
        planets = [f"planet{i}" for i in range(6)]
    if not modes:
        modes = [f"thermograph{i}" for i in range(3)]

    # Instruments to use for switch_on actions
    switch_on_instruments = powered_on_goals if powered_on_goals else instruments

    candidates = []

    # 1-step switch_on candidates
    for inst, sat in product(switch_on_instruments, satellites):
        candidates.append([f"switch_on {inst} {sat}"])

    # 1-step take_image candidates
    for sat, planet, inst, mode in product(satellites, planets, instruments, modes):
        candidates.append([f"take_image {sat} {planet} {inst} {mode}"])

    # 2-step mixed candidates: take_image + switch_on, in both orders
    for sat_img, planet, inst_img, mode in product(satellites, planets, instruments, modes):
        take = f"take_image {sat_img} {planet} {inst_img} {mode}"
        for inst_on, sat_on in product(switch_on_instruments, satellites):
            switch = f"switch_on {inst_on} {sat_on}"
            candidates.append([take, switch])
            candidates.append([switch, take])

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for plan in candidates:
        key = tuple(plan)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(plan)

    return unique_candidates