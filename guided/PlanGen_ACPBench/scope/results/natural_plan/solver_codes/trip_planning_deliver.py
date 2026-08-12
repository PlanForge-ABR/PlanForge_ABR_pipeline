def deliver_func(data):
    city_entries = [item for item in data if "city" in item]
    num_cities = len(city_entries)
    total_days = max(item["end_day"] for item in city_entries) if city_entries else 0

    lines = [
        f"Here is the trip plan for visiting the {num_cities} European cities for {total_days} days:",
        ""
    ]

    first_city = True
    for item in data:
        if "city" in item:
            city = item["city"]
            start_day = item["start_day"]
            end_day = item["end_day"]
            duration = end_day - start_day + 1
            if first_city:
                lines.append(
                    f"**Day {start_day}-{end_day}:** Arriving in {city} and visit {city} for {duration} days."
                )
                first_city = False
            else:
                lines.append(
                    f"**Day {start_day}-{end_day}:** Visit {city} for {duration} days."
                )
        elif "flight_day" in item:
            lines.append(
                f"**Day {item['flight_day']}:** Fly from {item['from']} to {item['to']}."
            )

    return "\n".join(lines)