"""Builder implementation of swap-domain methods."""

from typing import Dict, Iterable, List, Optional, Tuple

from architect.spec import SwapGoals, SwapState, SolveResult


def construct_plan(initial: SwapState, goals: SwapGoals) -> SolveResult:
    impossible = _static_impossibility(initial, goals)
    if impossible:
        return SolveResult(False, tuple(), impossible)
    if goals_hold(initial, goals):
        return SolveResult(True, tuple(), "goals already hold")

    current = dict(initial.assignments)
    plan: List[str] = []

    for agent, target_role in goals.atoms:
        if current[agent] == target_role:
            continue

        holder = _holder_of(current, target_role)
        if holder is None:
            return SolveResult(False, tuple(), f"role {target_role} is not currently assigned")
        if holder == agent:
            continue

        role_a = current[agent]
        role_b = current[holder]
        if role_a == role_b:
            return SolveResult(False, tuple(), "swap requires two distinct roles")

        plan.append(f"swap {agent} {holder} {role_a} {role_b}")
        current[agent], current[holder] = role_b, role_a

    final_state = _replace_assignments(initial, current)
    if not goals_hold(final_state, goals):
        return SolveResult(False, tuple(), "constructed swaps did not satisfy the requested facts")
    return SolveResult(True, tuple(plan), "consistent assignment repaired by swaps")


def goals_hold(state: SwapState, goals: SwapGoals) -> bool:
    assignments = dict(state.assignments)
    return all(assignments.get(agent) == role for agent, role in goals.atoms)


def simulate_plan(initial: SwapState, plan: Iterable[str]) -> SwapState:
    current = dict(initial.assignments)
    for action in plan:
        parts = action.split()
        if len(parts) != 5 or parts[0] != "swap":
            raise ValueError(f"unknown action: {action}")
        _, a1, a2, r1, r2 = parts
        if a1 == a2 or r1 == r2:
            raise ValueError(f"illegal self swap: {action}")
        if current.get(a1) != r1 or current.get(a2) != r2:
            raise ValueError(f"swap precondition failed: {action}")
        current[a1], current[a2] = r2, r1
    return _replace_assignments(initial, current)


def _static_impossibility(state: SwapState, goals: SwapGoals) -> Optional[str]:
    assignments = dict(state.assignments)
    agent_targets: Dict[str, set] = {}
    role_targets: Dict[str, set] = {}

    for agent, role in goals.atoms:
        if agent not in state.agents or agent not in assignments:
            return f"unknown agent {agent}"
        if role not in state.roles:
            return f"unknown role {role}"
        agent_targets.setdefault(agent, set()).add(role)
        role_targets.setdefault(role, set()).add(agent)

    for agent, roles in agent_targets.items():
        if len(roles) > 1:
            return f"{agent} cannot be assigned multiple roles at once"
    for role, agents in role_targets.items():
        if len(agents) > 1:
            return f"{role} cannot be assigned to multiple agents at once"

    if len(set(assignments.values())) != len(assignments):
        return "initial assignment is not one-to-one"
    return None


def _holder_of(assignments: Dict[str, str], role: str) -> Optional[str]:
    for agent, held_role in assignments.items():
        if held_role == role:
            return agent
    return None


def _replace_assignments(state: SwapState, assignments: Dict[str, str]) -> SwapState:
    return SwapState(
        assignments=tuple(sorted(assignments.items())),
        agents=state.agents,
        roles=state.roles,
    )
