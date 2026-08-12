def successor(given_state):
    """
    Generate all valid successor states from the given state in the ferry domain.
    
    Ferry domain actions:
    1. sail(from_loc, to_loc): Ferry moves between locations
    2. embark(car, location): Car boards the ferry at a location
    3. disembark(car, location): Car leaves the ferry at a location
    
    Args:
        given_state: List of predicates representing the current state
        
    Returns:
        List of (action, successor_state) pairs, where:
        - action is a string describing the action taken
        - successor_state is a list of predicates representing the next state
    """
    
    # Parse the current state into useful structures
    cars_at = {}  # car -> location
    ferry_at = None  # ferry location
    cars_on_ferry = set()  # set of cars on ferry
    locations = set()  # all known locations
    not_eq_pairs = set()  # pairs of different locations
    empty_ferry_present = False
    
    for pred in given_state:
        p_name = pred["predicate"]
        args = pred["args"]
        
        if p_name == "at":
            car, loc = args[0], args[1]
            cars_at[car] = loc
            locations.add(loc)
        elif p_name == "at-ferry":
            ferry_at = args[0]
            locations.add(ferry_at)
        elif p_name == "on":
            # on(car) or on(car, ferry)
            car = args[0]
            cars_on_ferry.add(car)
        elif p_name == "not-eq":
            loc1, loc2 = args[0], args[1]
            not_eq_pairs.add((loc1, loc2))
            locations.add(loc1)
            locations.add(loc2)
        elif p_name == "empty-ferry":
            empty_ferry_present = True
    
    successors = []
    
    # Helper function to create a new state with modifications
    def create_successor(add_preds, remove_preds):
        """Create a new state by adding and removing predicates."""
        # Convert to set of tuples for efficient operations
        state_set = set()
        for pred in given_state:
            pred_tuple = (pred["predicate"], tuple(pred["args"]))
            state_set.add(pred_tuple)
        
        # Remove predicates
        for pred in remove_preds:
            pred_tuple = (pred["predicate"], tuple(pred["args"]))
            state_set.discard(pred_tuple)
        
        # Add predicates
        for pred in add_preds:
            pred_tuple = (pred["predicate"], tuple(pred["args"]))
            state_set.add(pred_tuple)
        
        # Convert back to list of dicts and sort for consistency
        new_state = []
        for pred_tuple in state_set:
            new_state.append({
                "predicate": pred_tuple[0],
                "args": list(pred_tuple[1])
            })
        
        # Sort for canonical representation
        new_state.sort(key=lambda x: (x["predicate"], tuple(x["args"])))
        return new_state
    
    # Action 1: sail(from_loc, to_loc)
    # Preconditions: at-ferry(from_loc), not-eq(from_loc, to_loc)
    # Effects: at-ferry(to_loc), not at-ferry(from_loc)
    if ferry_at is not None:
        for to_loc in locations:
            if to_loc != ferry_at and (ferry_at, to_loc) in not_eq_pairs:
                add_preds = [{"predicate": "at-ferry", "args": [to_loc]}]
                remove_preds = [{"predicate": "at-ferry", "args": [ferry_at]}]
                action_str = f"sail {ferry_at} {to_loc}"
                successors.append((action_str, create_successor(add_preds, remove_preds)))
    
    # Action 2: embark(car, location)
    # Preconditions: at(car, loc), at-ferry(loc), not on(car), empty-ferry (ferry has capacity of 1)
    # Effects: on(car), not at(car, loc), not empty-ferry
    if ferry_at is not None and len(cars_on_ferry) == 0:
        # Ferry must be empty to embark a car
        for car, car_loc in cars_at.items():
            if car_loc == ferry_at and car not in cars_on_ferry:
                add_preds = [{"predicate": "on", "args": [car]}]
                remove_preds = [{"predicate": "at", "args": [car, car_loc]}]
                
                # If ferry was empty, it's no longer empty
                if empty_ferry_present:
                    remove_preds.append({"predicate": "empty-ferry", "args": []})

                action_str = f"board {car} {car_loc}"
                successors.append((action_str, create_successor(add_preds, remove_preds)))
    
    # Action 3: disembark(car, location)
    # Preconditions: on(car), at-ferry(loc)
    # Effects: at(car, loc), not on(car), empty-ferry (if this was last car)
    if ferry_at is not None:
        for car in cars_on_ferry:
            add_preds = [{"predicate": "at", "args": [car, ferry_at]}]
            remove_preds = [{"predicate": "on", "args": [car]}]
            
            # If this is the last car on ferry, ferry becomes empty
            if len(cars_on_ferry) == 1:
                add_preds.append({"predicate": "empty-ferry", "args": []})

            action_str = f"debark {car} {ferry_at}"
            successors.append((action_str, create_successor(add_preds, remove_preds)))
    
    return successors
