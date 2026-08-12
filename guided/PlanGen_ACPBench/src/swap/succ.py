"""
Successor function for the swap domain.

States are represented as lists of predicates, each expressed as a dict with
``predicate`` and ``args`` keys. Dynamic facts use the ``assigned(person, item)``
predicate while everything else (e.g., ``not-eq`` constraints) is static and
carried over unchanged to successor states.
"""
from typing import Dict, List, Tuple

Predicate = Dict[str, List[str]]
State = List[Predicate]


def _clone_fact(fact: Predicate) -> Predicate:
    """Return a shallow clone of a predicate dict."""
    return {
        "predicate": fact.get("predicate"),
        "args": list(fact.get("args", [])),
    }


def successor(state: State) -> List[Tuple[str, State]]:
    """
    Generate successor states by swapping the items assigned to every unordered
    pair of people. Each swap yields a new state where the pair's `assigned`
    facts are exchanged and the static predicates remain unchanged.
    """
    assignments: Dict[str, str] = {}
    order: List[str] = []
    static_facts: List[Predicate] = []

    for fact in state:
        predicate = fact.get("predicate")
        args = fact.get("args", [])
        if predicate == "assigned" and len(args) >= 2:
            person, item = args[0], args[1]
            if person not in assignments:
                order.append(person)
            assignments[person] = item
        else:
            static_facts.append(_clone_fact(fact))

    results: List[Tuple[str, State]] = []
    num_people = len(order)
    if num_people < 2:
        return results

    for i in range(num_people - 1):
        for j in range(i + 1, num_people):
            p1, p2 = order[i], order[j]
            new_assignment = assignments.copy()
            # Perform swap
            new_assignment[p1], new_assignment[p2] = new_assignment[p2], new_assignment[p1]
            
            # Reconstruct state
            new_state: State = []
            for person in order:
                new_state.append(
                    {
                        "predicate": "assigned",
                        "args": [person, new_assignment[person]],
                    }
                )
            for fact in static_facts:
                new_state.append(_clone_fact(fact))
            
            # Format action string: swap p1 p2 item1 item2
            # Note: The test expects specific ordering or just consistently formatted string
            # Looking at test data example: "action": "swap alice bob quince parsnip"
            # It seems to be: swap p1 p2 item_of_p1 item_of_p2 (before swap)
            item1 = assignments[p1]
            item2 = assignments[p2]
            action = f"swap {p1} {p2} {item1} {item2}"
            
            results.append((action, new_state))

    return results
