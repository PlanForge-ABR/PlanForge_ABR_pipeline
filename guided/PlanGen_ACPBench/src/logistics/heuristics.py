def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """Simple heuristic: count how many goal predicates are missing from state.

    This is admissible if each missing predicate requires at least one action.
    """
    sset = set((p['predicate'], tuple(p['args'])) for p in state)
    # Build location map
    obj_loc = {}
    in_city = {}
    # packages, trucks, airplanes
    packages = set()
    trucks = set()
    airplanes = set()

    # Parse state
    for p in state:
        pred = p['predicate']
        args = p['args']
        if pred == 'at':
            obj, loc = args
            obj_loc[obj] = loc
        elif pred == 'in':
            obj, vehicle = args
            obj_loc[obj] = vehicle 
        elif pred == 'in-city':
            loc, city = args
            in_city[loc] = city

    # Identify object types (heuristic guess based on names or usage)
    # This is imperfect without type info but we can infer:
    # if it's in a truck, it's a package. if it's 'at' a location, it could be anything.
    # We can infer from goal.
    
    total_cost = 0
    for g in goal:
        if (g['predicate'], tuple(g['args'])) in sset:
            continue
        
        pred = g['predicate']
        args = g['args']
        
        if pred == 'at':
            pkg, target_loc = args
            current_loc = obj_loc.get(pkg)
            
            if current_loc == target_loc:
                continue
            
            # If current_loc is None, we don't know where it is, assumed far
            if not current_loc:
                total_cost += 5
                continue

            # Check if package is in a vehicle
            # If current_loc is a vehicle (not in in_city), we need to unload it
            is_in_vehicle = current_loc not in in_city
            
            if is_in_vehicle:
                # Cost to unload + move + load etc.
                # Simplified: 1 (unload) + dist
                current_loc_of_vehicle = obj_loc.get(current_loc) # Where is the vehicle?
                if not current_loc_of_vehicle: 
                     total_cost += 5
                     continue
                start_city = in_city.get(current_loc_of_vehicle)
            else:
                start_city = in_city.get(current_loc)

            target_city = in_city.get(target_loc)

            if start_city == target_city:
                # Same city: Drive/Fly + Unload (if in vehicle)
                # Lower bound: 1 (load) + 1 (drive) + 1 (unload) = 3
                # If already in vehicle: 1 (drive) + 1 (unload) = 2
                if is_in_vehicle:
                    total_cost += 2
                else:
                    total_cost += 3
            else:
                # Different city: Load -> Drive -> Unload -> Load Plane -> Fly -> Unload Plane -> Load Truck -> Drive -> Unload
                # This is a bit much. Simplified:
                # Need to fly. 
                # If in vehicle: Unload(1) + LoadPlane(1) + Fly(1) + UnloadPlane(1) + LoadTruck(1) + Drive(1) + Unload(1) ~ 7
                total_cost += 5 # Heuristic guess
                
    return total_cost
