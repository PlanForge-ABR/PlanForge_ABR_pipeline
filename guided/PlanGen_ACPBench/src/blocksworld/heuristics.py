from collections import defaultdict
from typing import Dict, List, Any

def blocksworld9_cost(state_obj: Dict[str, Any], goals: List[Dict[str, Any]]) -> int:
    """
    Compute the BlocksWorld heuristic cost for the given state, using the
    same logic as the original blocksworld9Heuristic class.

    Parameters
    ----------
    state_obj : dict
        {
          "state": [
            {"predicate": "on",       "args": ["a", "b"]},
            {"predicate": "on-table", "args": ["c"]},
            {"predicate": "holding",  "args": ["d"]},  # optional
            ...
          ]
        }
    goals : list of dict
        {
          "state": [
          {"predicate": "on",       "args": ["a", "b"]},
          {"predicate": "on",       "args": ["b", "c"]},
          {"predicate": "on-table", "args": ["c"]},
          ...
          ]
        } 

    Returns
    -------
    int
        Heuristic cost estimate:
          - For each goal block not in its goal position:
              cost += 2 * (# of blocks above it + 1)
          - If a block is held and it has a goal position, add +1
    """
    # ---- parse goals -> goal_parent map ----
    goal_parent: Dict[str, str] = {}
    for g in goals:
        pred = g["predicate"]
        args = g["args"]
        if pred == "on":
            # (on child parent)
            child, parent = args[0], args[1]
            goal_parent[child] = parent
        elif pred == "on-table":
            # (on-table block)
            block = args[0]
            goal_parent[block] = "table"

    # ---- parse current state -> parent map, children map, held block ----
    current_parent: Dict[str, str] = {}
    current_children: Dict[str, List[str]] = defaultdict(list)
    held_block = None

    for f in state_obj.get("state", []):
        pred = f["predicate"]
        args = f["args"]
        if pred == "on":
            child, parent = args[0], args[1]
            current_parent[child] = parent
            current_children[parent].append(child)
        elif pred == "on-table":
            block = args[0]
            current_parent[block] = "table"
        elif pred == "holding":
            held_block = args[0]

    # ---- compute cost ----
    cost = 0

    # Held block: if it appears in the goal at all and is currently held,
    # it's not yet in its goal position → +1
    if held_block is not None and held_block in goal_parent:
        cost += 1

    # For each goal block: if not currently at its goal parent,
    # pay 2 * (#above + 1). Skip the held block since we already accounted for it.
    def count_above(x: str) -> int:
        cnt = 0
        stack = [x]
        while stack:
            cur = stack.pop()
            for c in current_children.get(cur, []):
                cnt += 1
                stack.append(c)
        return cnt

    for block, gparent in goal_parent.items():
        if block == held_block:
            continue
        cparent = current_parent.get(block, "table")
        if cparent != gparent:
            above = count_above(block)
            cost += 2 * (above + 1)

    return cost
