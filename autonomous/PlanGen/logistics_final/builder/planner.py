"""Builder implementation of the architect's logistics methods."""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from architect.spec import LogisticsGoals, LogisticsState, SolveResult


def construct_plan(initial: LogisticsState, goals: LogisticsGoals) -> SolveResult:
    ok, reason = validate_goal_consistency(initial, goals)
    if not ok:
        return SolveResult(False, [], reason)

    state = deepcopy(initial)
    plan: List[str] = []

    for package, target in sorted(goals.at.items(), key=lambda item: _object_key(item[0])):
        if package in state.packages:
            _deliver_package_to_location(state, plan, package, target)

    for package, vehicle in sorted(goals.in_vehicle.items(), key=lambda item: _object_key(item[0])):
        if state.in_vehicle.get(package) == vehicle:
            continue
        if vehicle in state.trucks:
            truck_loc = state.at[vehicle]
            _deliver_package_to_location(state, plan, package, truck_loc)
            _load_truck(state, plan, package, vehicle, truck_loc)
        else:
            plane_loc = state.at[vehicle]
            _deliver_package_to_location(state, plan, package, plane_loc)
            _load_airplane(state, plan, package, vehicle, plane_loc)

    for truck, target in sorted(goals.at.items(), key=lambda item: _object_key(item[0])):
        if truck in state.trucks and state.at.get(truck) != target:
            city = state.location_city[target]
            _drive_truck(state, plan, truck, target, city)

    for airplane, target in sorted(goals.at.items(), key=lambda item: _object_key(item[0])):
        if airplane in state.airplanes and state.at.get(airplane) != target:
            _fly_airplane(state, plan, airplane, target)

    return _finish(initial, goals, plan)


def validate_goal_consistency(state: LogisticsState, goals: LogisticsGoals) -> Tuple[bool, str]:
    if goals.object_location_conflict:
        return False, "one object cannot be at two distinct locations"
    if goals.object_container_conflict:
        return False, "one package cannot be inside two distinct vehicles"
    if goals.object_mixed_conflict:
        return False, "a package cannot be both at a location and inside a vehicle"

    for obj, loc in goals.at.items():
        if loc not in state.location_city:
            return False, f"unknown location {loc}"
        if obj in state.packages:
            continue
        if obj in state.trucks:
            current = state.at.get(obj)
            if current is None:
                return False, f"truck {obj} is not at any location"
            if state.location_city[current] != state.location_city[loc]:
                return False, "trucks cannot drive between cities"
            continue
        if obj in state.airplanes:
            if loc not in set(state.city_airport.values()):
                return False, "airplanes can only be at airport locations"
            continue
        return False, f"unknown object {obj}"

    for package, vehicle in goals.in_vehicle.items():
        if package not in state.packages:
            return False, f"unknown package {package}"
        if vehicle not in state.trucks and vehicle not in state.airplanes:
            return False, f"unknown vehicle {vehicle}"
    return True, ""


def simulate_plan(initial: LogisticsState, plan: List[str]) -> LogisticsState:
    state = deepcopy(initial)
    for action in plan:
        parts = action.split()
        if not parts:
            continue
        op = parts[0]
        if op == "DRIVE-TRUCK" and len(parts) == 5:
            _drive_truck(state, None, parts[1], parts[3], parts[4], expected_from=parts[2])
        elif op == "FLY-AIRPLANE" and len(parts) == 4:
            _fly_airplane(state, None, parts[1], parts[3], expected_from=parts[2])
        elif op == "LOAD-TRUCK" and len(parts) == 4:
            _load_truck(state, None, parts[1], parts[2], parts[3])
        elif op == "UNLOAD-TRUCK" and len(parts) == 4:
            _unload_truck(state, None, parts[1], parts[2], parts[3])
        elif op == "LOAD-AIRPLANE" and len(parts) == 4:
            _load_airplane(state, None, parts[1], parts[2], parts[3])
        elif op == "UNLOAD-AIRPLANE" and len(parts) == 4:
            _unload_airplane(state, None, parts[1], parts[2], parts[3])
        else:
            raise ValueError(f"bad action: {action}")
    return state


def goals_hold(state: LogisticsState, goals: LogisticsGoals) -> bool:
    for obj, loc in goals.at.items():
        if state.at.get(obj) != loc:
            return False
    for package, vehicle in goals.in_vehicle.items():
        if state.in_vehicle.get(package) != vehicle:
            return False
    return True


def _finish(initial: LogisticsState, goals: LogisticsGoals, plan: List[str]) -> SolveResult:
    try:
        final_state = simulate_plan(initial, plan)
    except Exception as exc:
        return SolveResult(False, [], f"generated plan failed simulation: {exc}")
    if not goals_hold(final_state, goals):
        return SolveResult(False, [], "constructive planner could not satisfy the requested facts")
    return SolveResult(True, plan, "")


def _deliver_package_to_location(
    state: LogisticsState, plan: List[str], package: str, target_loc: str
) -> None:
    if state.at.get(package) == target_loc:
        return

    if package in state.in_vehicle:
        vehicle = state.in_vehicle[package]
        if vehicle in state.trucks:
            vehicle_loc = state.at[vehicle]
            target_city = state.location_city[target_loc]
            if state.location_city[vehicle_loc] == target_city:
                _drive_truck(state, plan, vehicle, target_loc, target_city)
                _unload_truck(state, plan, package, vehicle, target_loc)
                return
            source_airport = _airport_for_location(state, vehicle_loc)
            _drive_truck(state, plan, vehicle, source_airport, state.location_city[vehicle_loc])
            _unload_truck(state, plan, package, vehicle, source_airport)
        else:
            plane_loc = state.at[vehicle]
            target_city = state.location_city[target_loc]
            target_airport = state.city_airport[target_city]
            _fly_airplane(state, plan, vehicle, target_airport)
            _unload_airplane(state, plan, package, vehicle, target_airport)

    current_loc = state.at[package]
    current_city = state.location_city[current_loc]
    target_city = state.location_city[target_loc]
    if current_city == target_city:
        truck = _truck_for_city(state, current_city)
        _move_package_with_truck(state, plan, package, truck, target_loc)
        return

    source_airport = state.city_airport[current_city]
    target_airport = state.city_airport[target_city]
    source_truck = _truck_for_city(state, current_city)
    if current_loc != source_airport:
        _move_package_with_truck(state, plan, package, source_truck, source_airport)

    plane = _first_airplane(state)
    _fly_airplane(state, plan, plane, source_airport)
    _load_airplane(state, plan, package, plane, source_airport)
    _fly_airplane(state, plan, plane, target_airport)
    _unload_airplane(state, plan, package, plane, target_airport)

    if target_loc != target_airport:
        target_truck = _truck_for_city(state, target_city)
        _move_package_with_truck(state, plan, package, target_truck, target_loc)


def _move_package_with_truck(
    state: LogisticsState, plan: List[str], package: str, truck: str, target_loc: str
) -> None:
    source_loc = state.at[package]
    city = state.location_city[source_loc]
    _drive_truck(state, plan, truck, source_loc, city)
    _load_truck(state, plan, package, truck, source_loc)
    _drive_truck(state, plan, truck, target_loc, city)
    _unload_truck(state, plan, package, truck, target_loc)


def _drive_truck(
    state: LogisticsState,
    plan: Optional[List[str]],
    truck: str,
    to_loc: str,
    city: str,
    expected_from: Optional[str] = None,
) -> None:
    from_loc = state.at.get(truck)
    if from_loc is None:
        raise ValueError(f"{truck} is not at a location")
    if expected_from is not None and from_loc != expected_from:
        raise ValueError(f"{truck} is not at {expected_from}")
    if state.location_city[from_loc] != city or state.location_city[to_loc] != city:
        raise ValueError(f"{truck} cannot drive outside {city}")
    if from_loc == to_loc:
        return
    state.at[truck] = to_loc
    if plan is not None:
        plan.append(f"DRIVE-TRUCK {truck} {from_loc} {to_loc} {city}")


def _fly_airplane(
    state: LogisticsState,
    plan: Optional[List[str]],
    airplane: str,
    to_loc: str,
    expected_from: Optional[str] = None,
) -> None:
    from_loc = state.at.get(airplane)
    if from_loc is None:
        raise ValueError(f"{airplane} is not at a location")
    airports = set(state.city_airport.values())
    if from_loc not in airports or to_loc not in airports:
        raise ValueError(f"{airplane} can only fly between airports")
    if expected_from is not None and from_loc != expected_from:
        raise ValueError(f"{airplane} is not at {expected_from}")
    if from_loc == to_loc:
        return
    state.at[airplane] = to_loc
    if plan is not None:
        plan.append(f"FLY-AIRPLANE {airplane} {from_loc} {to_loc}")


def _load_truck(
    state: LogisticsState, plan: Optional[List[str]], package: str, truck: str, loc: str
) -> None:
    if state.at.get(package) != loc or state.at.get(truck) != loc:
        raise ValueError(f"{package} and {truck} are not both at {loc}")
    del state.at[package]
    state.in_vehicle[package] = truck
    if plan is not None:
        plan.append(f"LOAD-TRUCK {package} {truck} {loc}")


def _unload_truck(
    state: LogisticsState, plan: Optional[List[str]], package: str, truck: str, loc: str
) -> None:
    if state.in_vehicle.get(package) != truck or state.at.get(truck) != loc:
        raise ValueError(f"{package} is not in {truck} at {loc}")
    del state.in_vehicle[package]
    state.at[package] = loc
    if plan is not None:
        plan.append(f"UNLOAD-TRUCK {package} {truck} {loc}")


def _load_airplane(
    state: LogisticsState, plan: Optional[List[str]], package: str, airplane: str, loc: str
) -> None:
    if state.at.get(package) != loc or state.at.get(airplane) != loc:
        raise ValueError(f"{package} and {airplane} are not both at {loc}")
    del state.at[package]
    state.in_vehicle[package] = airplane
    if plan is not None:
        plan.append(f"LOAD-AIRPLANE {package} {airplane} {loc}")


def _unload_airplane(
    state: LogisticsState, plan: Optional[List[str]], package: str, airplane: str, loc: str
) -> None:
    if state.in_vehicle.get(package) != airplane or state.at.get(airplane) != loc:
        raise ValueError(f"{package} is not in {airplane} at {loc}")
    del state.in_vehicle[package]
    state.at[package] = loc
    if plan is not None:
        plan.append(f"UNLOAD-AIRPLANE {package} {airplane} {loc}")


def _truck_for_city(state: LogisticsState, city: str) -> str:
    for truck in sorted(state.trucks, key=_object_key):
        loc = state.at.get(truck)
        if loc is not None and state.location_city[loc] == city:
            return truck
    raise ValueError(f"no truck available in {city}")


def _airport_for_location(state: LogisticsState, loc: str) -> str:
    return state.city_airport[state.location_city[loc]]


def _first_airplane(state: LogisticsState) -> str:
    if not state.airplanes:
        raise ValueError("no airplane available")
    return sorted(state.airplanes, key=_object_key)[0]


def _object_key(name: str):
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return (prefix, int(digits) if digits else -1)
