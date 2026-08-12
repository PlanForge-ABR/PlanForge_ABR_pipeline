from typing import Dict, List, Sequence, Tuple

Predicate = Tuple[str, Tuple[str, ...]]
State = List[Dict[str, Sequence[str]]]


def _to_predicate(entry: Dict[str, Sequence[str]]) -> Predicate:
    return entry.get("predicate"), tuple(entry.get("args", []) or [])


def is_goal(current_state: State, goal_state: State) -> bool:
    """Return True when every goal predicate appears in the current state."""
    state_facts = {_to_predicate(p) for p in current_state}
    for goal in goal_state:
        if _to_predicate(goal) not in state_facts:
            return False
    return True
