"""
Goal test for the Satellite domain.

The goal is satisfied when every fact listed in the goal_state appears in the
current_state.  Both states use the {"predicate": str, "args": [...]}
representation.
"""
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Fact = Tuple[str, Tuple[str, ...]]
State = List[Dict[str, Sequence[str]]]


def _to_fact_set(facts: Iterable[Dict[str, Sequence[str]]]) -> Set[Fact]:
    return {
        (fact.get("predicate"), tuple(fact.get("args", ())))
        for fact in facts
    }


def is_goal(current_state: State, goal_state: State) -> bool:
    """
    Check whether all goal facts exist in the current state.
    Missing or empty goal states default to False/True respectively.
    """
    goal_state = goal_state or []
    if not goal_state:
        return True

    current_facts = _to_fact_set(current_state or [])
    for goal_fact in goal_state:
        predicate = goal_fact.get("predicate")
        args = tuple(goal_fact.get("args", ()))
        if (predicate, args) not in current_facts:
            return False
    return True
