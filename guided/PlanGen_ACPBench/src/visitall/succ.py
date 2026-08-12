from typing import List, Dict, Any, Tuple

Predicate = Dict[str, List[str]]

def _apply_effects(state: List[Predicate],
                   add_effects: List[Predicate],
                   del_effects: List[Predicate]) -> List[Predicate]:
    """Apply add/delete effects over set semantics and return a sorted state."""
    state_tuples = {(p["predicate"], tuple(p["args"])) for p in state}
    del_tuples = {(p["predicate"], tuple(p["args"])) for p in del_effects}
    add_tuples = {(p["predicate"], tuple(p["args"])) for p in add_effects}

    new_tuples = (state_tuples - del_tuples) | add_tuples

    # Deterministic ordering of predicates inside each state
    return sorted(
        ({"predicate": pred, "args": list(args)} for pred, args in new_tuples),
        key=lambda x: (x["predicate"], x["args"])
    )

def successor(state: List[Predicate]) -> List[Tuple[str, List[Predicate]]]:
    """
    Successor generator for the VisitAll (grid) domain.

    Input state facts (examples):
      {'predicate': 'at-robot', 'args': ['loc-x2-y1']}
      {'predicate': 'connected', 'args': ['loc-x2-y1', 'loc-x3-y1']}
      {'predicate': 'visited', 'args': ['loc-x1-y1']}

    Transition:
      If at-robot(L) and connected(L, N), then you can move to N:
        - delete at-robot(L), add at-robot(N)
        - ensure visited(N) is present

    Returns:
      A list of successor states, each a list of predicate dicts.
    """
    # Find current robot location
    curr_loc = None
    for fact in state:
        if fact.get("predicate") == "at-robot":
            curr_loc = fact["args"][0]
            break
    if curr_loc is None:
        # No robot in state -> no successors
        return []

    # Collect directed neighbors from connected facts
    neighbors = []
    for fact in state:
        if fact.get("predicate") == "connected":
            src, dst = fact["args"][0], fact["args"][1]
            if src == curr_loc:
                neighbors.append(dst)

    # Unique, deterministic neighbor list
    neighbors = sorted(set(neighbors))

    # Build successors
    succ_states: List[Tuple[str, List[Predicate]]] = []
    for nxt in neighbors:
        add = [
            {"predicate": "at-robot", "args": [nxt]},
            {"predicate": "visited", "args": [nxt]},
        ]
        dels = [
            {"predicate": "at-robot", "args": [curr_loc]},
        ]
        
        new_state = _apply_effects(state, add, dels)
        action = f"move {curr_loc} {nxt}"
        succ_states.append((action, new_state))

    # Deterministic ordering of successor list by action string (since destination is in action)
    succ_states.sort(key=lambda x: x[0])

    # Deduplicate identical states (e.g., duplicate connected facts)
    # We need to preserve the action associated with the unique state.
    # If multiple actions lead to same state (unlikely here but possible in general), we take first?
    # In visitall, move A->B is unique given state has Robot at A.
    unique = {}
    for action, s in succ_states:
        key = frozenset((p["predicate"], tuple(p["args"])) for p in s)
        if key not in unique:
             unique[key] = (action, s)
    
    # Return list of (action, state) tuples
    return list(unique.values())
