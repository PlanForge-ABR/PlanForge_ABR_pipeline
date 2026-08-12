from typing import Any, Dict, Iterable, List, Tuple


Predicate = Dict[str, Any]


def _normalize(predicates: Iterable[Predicate]) -> List[Tuple[str, Tuple[str, ...]]]:
    """Convert predicate dicts to sortable tuples."""
    normalized: List[Tuple[str, Tuple[str, ...]]] = []
    for pred in predicates:
        name = pred.get("predicate")
        if not name:
            continue
        args = pred.get("args", [])
        if not isinstance(args, (list, tuple)):
            args = [args]
        normalized.append((name, tuple(args)))
    normalized.sort()
    return normalized


def is_goal(current_state: List[Predicate], goal_state: List[Predicate]) -> bool:
    """
    Return True when every predicate in ``goal_state`` is present in ``current_state``.

    Both inputs use the common representation of a state as a list of dictionaries
    with ``predicate`` and ``args`` keys.  The goal can be a partial description,
    so satisfaction requires the goal predicates to be a subset of the current state.
    """
    current = set(_normalize(current_state))
    goal = _normalize(goal_state)
    return all(g in current for g in goal)
