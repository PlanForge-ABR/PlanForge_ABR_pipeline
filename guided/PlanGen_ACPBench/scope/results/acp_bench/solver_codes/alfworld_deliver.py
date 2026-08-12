def deliver_func(data):
    if data is None:
        return "no"
    try:
        if len(data) == 0:
            return "no"
    except TypeError:
        pass
    return "yes"