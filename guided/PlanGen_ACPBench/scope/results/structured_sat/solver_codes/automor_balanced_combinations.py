def combinations_func(data):
    n = int(data.get("variables", 0))
    if n < 0:
        raise ValueError("data['variables'] must be a non-negative integer")

    return {
        "variable_order": [str(i) for i in range(1, n + 1)],
        "domain": (False, True),
        "count": n,
    }