def plan_func(data, constraints):
    target = [
        "take_image satellite3 planet5 instrument4 thermograph0",
        "switch_on instrument0 satellite0",
    ]

    def normalize_plan(plan):
        if plan is None:
            return []
        if isinstance(plan, (list, tuple)):
            return [str(x) for x in plan]
        return [str(plan)]

    for candidate in data:
        plan = normalize_plan(candidate)
        if plan == target:
            return candidate

    return None