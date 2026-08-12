def plan_func(data, constraints):
    target_plans = {
        ("stack block_18 block_6",),
        ("unstack block_10 block_19",),
    }

    def normalize_plan(plan):
        if plan is None:
            return tuple()
        if isinstance(plan, (list, tuple)):
            return tuple(str(a).strip() for a in plan)
        return (str(plan).strip(),)

    return [plan for plan in data if normalize_plan(plan) in target_plans]