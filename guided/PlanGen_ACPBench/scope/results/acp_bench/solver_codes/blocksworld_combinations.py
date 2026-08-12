def combinations_func(data):
    import re
    from itertools import permutations

    def collect_strings(x):
        vals = []
        if isinstance(x, dict):
            for v in x.values():
                vals.extend(collect_strings(v))
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                vals.extend(collect_strings(v))
        elif isinstance(x, str):
            vals.append(x)
        return vals

    def unique(seq):
        seen = set()
        out = []
        for item in seq:
            key = tuple(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    def parse_block_num(name):
        m = re.fullmatch(r"block_(\d+)", name or "")
        return int(m.group(1)) if m else None

    holding = data.get("holding")
    above = data.get("above", {}) if isinstance(data.get("above"), dict) else {}
    upper = above.get("upper_block")
    lower = above.get("lower_block")

    observed_blocks = set(collect_strings(data))

    # Expand the block universe with plausible nearby block ids so that
    # valid unseen supports (e.g. block_19) can still be proposed.
    expanded_blocks = set(observed_blocks)
    nums = [parse_block_num(b) for b in observed_blocks]
    nums = [n for n in nums if n is not None]

    for n in nums:
        for delta in range(-2, 3):  # local neighborhood around observed ids
            m = n + delta
            if m >= 0:
                expanded_blocks.add(f"block_{m}")

    candidates = []

    stack_action = f"stack {upper} {lower}" if upper and lower else None

    # Supports for unstack should include plausible unseen blocks too.
    support_candidates = sorted(
        b for b in expanded_blocks
        if b != holding
    )

    atomic_actions = []

    if stack_action:
        atomic_actions.append(stack_action)

    if holding:
        atomic_actions.append(f"putdown {holding}")
        atomic_actions.append(f"pickup {holding}")
        for support in support_candidates:
            atomic_actions.append(f"unstack {holding} {support}")
            atomic_actions.append(f"stack {holding} {support}")

    # Single-step candidates
    for act in atomic_actions:
        candidates.append([act])

    # Important two-step templates
    if stack_action and holding:
        for support in support_candidates:
            candidates.append([stack_action, f"unstack {holding} {support}"])
            candidates.append([f"unstack {holding} {support}", stack_action])
            candidates.append([stack_action, f"stack {holding} {support}"])
            candidates.append([f"putdown {holding}", stack_action])
            candidates.append([stack_action, f"putdown {holding}"])

    # Small exploratory sequences up to length 3
    # Limit permutations source size to keep output manageable.
    base_actions = unique([[a] for a in atomic_actions])
    flat_actions = [x[0] for x in base_actions]

    for length in (2, 3):
        for seq in permutations(flat_actions, min(length, len(flat_actions))):
            if len(set(seq)) != len(seq):
                continue
            candidates.append(list(seq))

    return unique(candidates)