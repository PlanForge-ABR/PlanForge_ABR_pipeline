def combinations_func(data):
    from itertools import product

    object_type = data["object_type"]
    receptacle_type = data["receptacle_type"]

    object_type_token = f"{object_type}type"
    receptacle_type_token = f"{receptacle_type}type"

    object_candidates = [f"{object_type}{i}" for i in range(1, 4)]
    receptacle_candidates = [f"{receptacle_type}{i}" for i in range(1, 4)]

    single_step_plans = []
    for obj, rec in product(object_candidates, receptacle_candidates):
        action = (
            f"validate_pick_and_place_in_receptacle "
            f"{obj} {object_type_token} {rec} {receptacle_type_token}"
        )
        single_step_plans.append([action])

    two_step_plans = []
    for (obj1, rec1), (obj2, rec2) in product(
        product(object_candidates, receptacle_candidates),
        product(object_candidates, receptacle_candidates),
    ):
        plan = [
            f"validate_pick_and_place_in_receptacle "
            f"{obj1} {object_type_token} {rec1} {receptacle_type_token}",
            f"validate_pick_and_place_in_receptacle "
            f"{obj2} {object_type_token} {rec2} {receptacle_type_token}",
        ]
        two_step_plans.append(plan)

    return single_step_plans + two_step_plans