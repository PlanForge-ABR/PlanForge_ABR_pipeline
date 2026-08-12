def deliver_func(data):
    itinerary = data[0] if isinstance(data, list) and data else data
    result = []

    start = itinerary.get("start", {})
    result.append(f'You start at {start.get("location")} at {start.get("time")}.')

    for step in itinerary.get("steps", []):
        action = step.get("action")
        if action == "travel":
            result.append(
                f'You travel to {step.get("to")} in {step.get("duration_minutes")} minutes and arrive at {step.get("arrival_time")}.'
            )
        elif action == "wait":
            result.append(f'You wait until {step.get("end_time")}.')
        elif action == "meet":
            result.append(
                f'You meet {step.get("person")} for {step.get("duration_minutes")} minutes from {step.get("start_time")} to {step.get("end_time")}.'
            )

    return result