from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


Fact = Tuple[str, Tuple[str, ...]]


def _normalize_facts(items: Iterable[Dict[str, Any]]) -> Set[Fact]:
    """Normalize a list of fact dicts into hashable (predicate, args) tuples."""
    out: Set[Fact] = set()
    for f in items:
        pred = f.get("predicate")
        args = f.get("args", [])
        if pred is None:
            continue
        if not isinstance(args, (list, tuple)):
            args = [args]
        out.add((str(pred), tuple(str(a) for a in args)))
    return out


def _extract_state(obj: Any) -> Sequence[Dict[str, Any]]:
    """Extract the underlying state list from either a dict or raw list."""
    if isinstance(obj, dict):
        state = obj.get("state", [])
        if isinstance(state, (list, tuple)):
            return state
        return []
    if isinstance(obj, (list, tuple)):
        return obj
    return []


def is_goal(current_state: Any, goal_state: Any) -> bool:
    """Return True if all goal predicates are present in the current state.

    Both inputs may be either:
      - a bare list of {"predicate": str, "args": [str, ...]} dicts,
      - a dict with key "state" mapping to such a list, or
      - a tuple of (pred, args) facts (optimized).
    """
    current_items = _extract_state(current_state)
    goal_items = _extract_state(goal_state)

    # Optimization: if already tuple of facts
    if isinstance(current_items, tuple):
        current_facts = set(current_items)
    else:
        current_facts = _normalize_facts(current_items)

    if isinstance(goal_items, tuple):
        goal_facts = set(goal_items)
    else:
        goal_facts = _normalize_facts(goal_items)

    return goal_facts.issubset(current_facts)

