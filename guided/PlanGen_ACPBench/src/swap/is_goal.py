"""
Goal test for the swap domain.

The goal is satisfied when every predicate present in the goal list also occurs
in the current state. Goals only describe the required portion of the final
assignment, so extra facts in the current state are allowed.
"""
from typing import Dict, Iterable, List

Predicate = Dict[str, List[str]]
State = List[Predicate]


def _to_fact_set(predicates: Iterable[Predicate]) -> set:
    """Represent predicates as hashable tuples for fast membership checks."""
    fact_set = set()
    for fact in predicates:
        predicate = fact.get("predicate")
        args = tuple(fact.get("args", []))
        fact_set.add((predicate, args))
    return fact_set


def is_goal(current_state: State, goal_state: State) -> bool:
    """
    Return True iff every goal predicate appears in the current state.
    """
    current_facts = _to_fact_set(current_state)
    for goal_fact in goal_state:
        predicate = goal_fact.get("predicate")
        args = tuple(goal_fact.get("args", []))
        if (predicate, args) not in current_facts:
            return False
    return True
