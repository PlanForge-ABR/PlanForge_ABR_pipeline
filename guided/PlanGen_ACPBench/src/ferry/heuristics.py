from typing import Dict, List, Any

def ferry_cost(state_obj: Dict[str, Any], goals: List[Dict[str, Any]]) -> int:
    """
    Compute the Ferry domain heuristic cost for the given state.
    
    This heuristic estimates the minimum number of actions needed to achieve
    the goal state from the current state.
    
    Parameters
    ----------
    state_obj : dict
        {
          "state": [
            {"predicate": "at",       "args": ["car1", "loc0"]},
            {"predicate": "at-ferry", "args": ["loc1"]},
            {"predicate": "on",       "args": ["car2"]},
            ...
          ]
        }
    goals : list of dict
        [
          {"predicate": "at", "args": ["car1", "loc2"]},
          {"predicate": "at", "args": ["car2", "loc3"]},
          ...
        ]
    
    Returns
    -------
    int
        Heuristic cost estimate (admissible - never overestimates)
        
    Heuristic Logic:
    ----------------
    For each car that needs to be moved to a different location:
      - If the car is already at its goal location: cost += 0
      - If the car is on the ferry and ferry needs to reach goal: cost += 1 (disembark)
      - If the car is not at goal and not on ferry: cost += 2 (embark + disembark minimum)
    
    This is admissible because:
      - Moving a car requires at minimum: embark (1) + disembark (1) = 2 actions
      - If car is already on ferry: disembark (1) = 1 action
      - Sail actions are not counted directly but are implicit in the transport
    
    Note: This is a simplified heuristic that doesn't account for:
      - Ferry capacity constraints (ferry can only hold 1 car)
      - Ferry repositioning between cars
      - Optimal sequencing of car transport
    Therefore it may underestimate the true cost, making it admissible.
    """
    
    # Parse current state
    current_cars_at = {}  # car -> location
    ferry_location = None
    cars_on_ferry = set()
    
    for predicate in state_obj.get("state", []):
        pred_name = predicate["predicate"]
        args = predicate["args"]
        
        if pred_name == "at":
            car, location = args[0], args[1]
            current_cars_at[car] = location
        elif pred_name == "at-ferry":
            ferry_location = args[0]
        elif pred_name == "on":
            car = args[0]
            cars_on_ferry.add(car)
    
    # Parse goal state - extract where each car needs to be
    goal_cars_at = {}  # car -> goal_location
    for goal_pred in goals:
        if goal_pred["predicate"] == "at":
            car, location = goal_pred["args"][0], goal_pred["args"][1]
            goal_cars_at[car] = location
    
    # Calculate heuristic cost
    cost = 0
    
    for car, goal_location in goal_cars_at.items():
        # Check current location of this car
        current_location = current_cars_at.get(car)
        
        if car in cars_on_ferry:
            # Car is on the ferry
            if ferry_location == goal_location:
                # Ferry is at goal location, just need to disembark
                cost += 1  # disembark
            else:
                # Ferry needs to sail to goal, then disembark
                # Minimum: 1 sail + 1 disembark = 2 actions
                # But we underestimate by counting only disembark
                cost += 1  # disembark (sail is implicit)
        elif current_location == goal_location:
            # Car is already at goal location
            cost += 0
        else:
            # Car needs to be transported
            # Minimum: embark + disembark = 2 actions
            # (sail actions are implicit and not counted separately)
            cost += 2  # embark + disembark
    
    return cost

