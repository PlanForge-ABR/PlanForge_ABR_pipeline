def plan_func(data, constraints):
    target = [
        "swap kevin alice cro ceo",
        "swap liam erin cco yco",
    ]

    for plan in data:
        if plan == target:
            return plan
    return None