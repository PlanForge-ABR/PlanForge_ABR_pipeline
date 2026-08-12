def combinations_func(data):
    """
    Return a search-space configuration for boolean assignment generation.

    The result is a dictionary containing the ordered variable identifiers
    to be assigned by the solver's backtracking logic, avoiding eager
    construction of all 2^N truth assignments.
    """
    if data is None:
        data = {}

    variable_ids = None

    if isinstance(data, dict):
        if "variable_ids" in data and isinstance(data["variable_ids"], (list, tuple)):
            variable_ids = [str(v) for v in data["variable_ids"]]
        elif "variables" in data:
            vars_field = data["variables"]
            if isinstance(vars_field, int):
                variable_ids = [str(i) for i in range(1, vars_field + 1)]
            elif isinstance(vars_field, dict):
                try:
                    variable_ids = sorted((str(k) for k in vars_field.keys()), key=lambda x: int(x))
                except Exception:
                    variable_ids = sorted(str(k) for k in vars_field.keys())
            elif isinstance(vars_field, (list, tuple)):
                variable_ids = [str(v) for v in vars_field]
        elif "num_variables" in data and isinstance(data["num_variables"], int):
            variable_ids = [str(i) for i in range(1, data["num_variables"] + 1)]
        elif "n" in data and isinstance(data["n"], int):
            variable_ids = [str(i) for i in range(1, data["n"] + 1)]

    if variable_ids is None:
        variable_ids = []

    # Deduplicate while preserving order.
    seen = set()
    ordered_ids = []
    for vid in variable_ids:
        if vid not in seen:
            seen.add(vid)
            ordered_ids.append(vid)

    return {
        "type": "variable_order",
        "variables": ordered_ids,
        "domain": (False, True),
    }