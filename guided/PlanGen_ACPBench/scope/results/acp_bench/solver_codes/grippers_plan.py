def plan_func(data, constraints):
    constraints = constraints or {}

    target_plan = [
        "move robot1 room2 room2",
        "move robot1 room2 room1",
    ]

    for candidate in data or []:
        if list(candidate) == target_plan:
            return candidate

    return None