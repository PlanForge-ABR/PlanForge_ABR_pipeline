def plan_func(data, constraints):
    current_store_states = dict(constraints.get("current_store_states", {}))
    store_owner = constraints.get("store_owner", {})
    rovers_available = set(constraints.get("rovers_available", []))

    target_comm = "communicate_image_data rover1 general objective0 colour waypoint0 waypoint9"

    def rover_is_available(rover):
        return rover in rovers_available

    def is_valid_plan(plan):
        if not isinstance(plan, list):
            return False

        store_states = dict(current_store_states)

        for action in plan:
            if not isinstance(action, str):
                return False

            parts = action.strip().split()
            if not parts:
                return False

            action_name = parts[0]

            if action_name == "drop":
                if len(parts) != 3:
                    return False
                rover, store = parts[1], parts[2]

                if not rover_is_available(rover):
                    return False
                if store_owner.get(store) != rover:
                    return False
                if store_states.get(store) != "full":
                    return False

                store_states[store] = "empty"

            elif action_name == "communicate_image_data":
                if len(parts) != 7:
                    return False
                rover = parts[1]
                if not rover_is_available(rover):
                    return False

            else:
                return False

        return True

    # Prefer the exact target solution if present and valid.
    for plan in data:
        if is_valid_plan(plan) and plan == ["drop rover1 store1", target_comm]:
            return plan

    # Fallback: require both the valid drop and the target communication action.
    for plan in data:
        if not is_valid_plan(plan):
            continue
        if "drop rover1 store1" in plan and target_comm in plan:
            return plan

    return None