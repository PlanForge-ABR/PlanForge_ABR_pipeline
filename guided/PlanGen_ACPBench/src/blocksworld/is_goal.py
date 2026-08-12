from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def _normalize_facts(items: Iterable[Dict[str, Any]]) -> Set[Tuple[str, Tuple[str, ...]]]:
    """Convert a list of fact dicts to a set of canonical tuples.

    Input item example: {"predicate": "on", "args": ["a", "b"]}
    Output tuple: ("on", ("a", "b"))
    """
    out: Set[Tuple[str, Tuple[str, ...]]] = set()
    for f in items:
        pred = f.get("predicate")
        args = f.get("args", [])
        if not isinstance(args, (list, tuple)):
            # Be forgiving: if a single value is provided, wrap it
            args = [args]
        out.add((str(pred), tuple(str(a) for a in args)))
    return out


def _derive_clear(facts: Set[Tuple[str, Tuple[str, ...]]]) -> Set[Tuple[str, Tuple[str, ...]]]:
    """Derive clear(x) facts from on(child, parent) relations.

    A block is clear if no other block is on it. This derivation augments
    any explicit clear facts present in the state.
    """
    # Collect all blocks that appear anywhere
    blocks: Set[str] = set()
    parents_with_children: Set[str] = set()
    for pred, args in facts:
        if pred == "on" and len(args) == 2:
            child, parent = args
            blocks.add(child)
            blocks.add(parent)
            parents_with_children.add(parent)
        elif pred in {"on-table", "ontable", "clear", "holding"} and len(args) >= 1:
            blocks.add(args[0])

    # Anything that is a block and is not a parent in any on(_, x) is clear
    derived: Set[Tuple[str, Tuple[str, ...]]] = set()
    for b in blocks:
        if b not in parents_with_children:
            derived.add(("clear", (b,)))
    return facts | derived


def is_goal(state_obj: Any, goals: Any) -> bool:
    """Return True if all goal facts hold in the given state.

    Parameters
    ----------
    state_obj: dict with key "state" mapping to a list of fact dicts.
    goals: either a list of fact dicts, or a dict with key "state".

    Notes
    -----
    - Supports predicates: on(a,b), on-table(a), holding(a), handempty(), clear(a).
    - "clear(x)" is also derived from on/holding relationships for robustness.
    """
    # Normalize inputs
    if isinstance(state_obj, dict):
        state_facts_raw: Sequence[Dict[str, Any]] = state_obj.get("state", [])
    else:
        state_facts_raw = state_obj or []
    goal_items: Sequence[Dict[str, Any]]
    if isinstance(goals, dict):
        goal_items = goals.get("state", [])
    else:
        goal_items = goals or []

    state_facts = _normalize_facts(state_facts_raw)

    # Augment with derived clear facts
    state_facts = _derive_clear(state_facts)

    goal_facts = _normalize_facts(goal_items)

    # All goal facts must be present in (possibly augmented) state facts
    return goal_facts.issubset(state_facts)
