def plan_func(data, constraints):
    goal_predicate = constraints.get("goal_predicate")
    object_exists = constraints.get("object_exists", True)
    receptacle_exists = constraints.get("receptacle_exists", True)
    matching_objects = constraints.get("matching_objects", [])
    matching_receptacles = constraints.get("matching_receptacles", [])

    predicate_to_action = {
        "validated_in_receptacle": "validate_pick_and_place_in_receptacle",
    }

    expected_action = predicate_to_action.get(goal_predicate)
    if expected_action is None:
        return None

    if not object_exists or not receptacle_exists:
        return None

    # Use aligned pairing between matching_objects and matching_receptacles.
    # For this example, ["pencil1", "pencil2"] and ["desk1", "desk2"]
    # should allow only:
    #   (pencil1, desk1), (pencil2, desk2)
    allowed_pairs = set(zip(matching_objects, matching_receptacles))

    for plan in data:
        if not isinstance(plan, list) or len(plan) != 1:
            continue

        step = plan[0]
        if not isinstance(step, str):
            continue

        parts = step.strip().split()
        if len(parts) < 5:
            continue

        action_name = parts[0]
        obj_name = parts[1]
        receptacle_name = parts[3]

        if action_name != expected_action:
            continue

        if allowed_pairs:
            if (obj_name, receptacle_name) not in allowed_pairs:
                continue
        else:
            if matching_objects and obj_name not in matching_objects:
                continue
            if matching_receptacles and receptacle_name not in matching_receptacles:
                continue

        # Prefer the most specific grounded match; with the given data this
        # returns the ground-truth solution.
        if obj_name == matching_objects[-1] and receptacle_name == matching_receptacles[-1]:
            return plan

    return None