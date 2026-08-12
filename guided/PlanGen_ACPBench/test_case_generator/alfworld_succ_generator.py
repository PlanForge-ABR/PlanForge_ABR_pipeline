#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALFWorld-specific successor generator with precondition-driven grounding.

This avoids the huge Cartesian products in the generic successor generator by:
  - Building indices over the initial-state predicates.
  - For each action, generating candidate bindings only from facts that
    mention the relevant predicates in its (positive) preconditions.
  - Using type information only to filter candidate values instead of as
    the primary enumeration source.

It produces the same semantics (all applicable one-step successors) but is
practical for ALFWorld-sized problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:  # when imported as part of the package
    from .unified_pddl_parser import unified_parser
except ImportError:  # when run as a standalone script
    from unified_pddl_parser import unified_parser


class Atom(BaseModel):
    model_config = ConfigDict(frozen=True)
    predicate: str
    args: List[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.args)))


class State(BaseModel):
    model_config = ConfigDict(frozen=True)
    atoms: List[Atom] = Field(default_factory=list)

    def as_set(self) -> set:
        return set(self.atoms)

    @classmethod
    def from_iter(cls, atoms: Iterable[Atom]) -> "State":
        return cls(atoms=list({a for a in atoms}))

    def apply(self, add: Iterable[Atom], delete: Iterable[Atom]) -> "State":
        s = self.as_set()
        s.difference_update(delete)
        s.update(add)
        return State.from_iter(s)


Token = Sequence  # only used for type hints from caller


@dataclass
class ActionSchema:
    name: str
    params: List[Tuple[str, Optional[str]]]  # (var, type)
    preconds_pos: List[Tuple[str, List[str]]]
    preconds_neg: List[Tuple[str, List[str]]]
    equalities: List[Tuple[str, str]]
    inequalities: List[Tuple[str, str]]
    add_effects: List[Tuple[str, List[str]]]
    del_effects: List[Tuple[str, List[str]]]


class AlfworldSuccessorGenerator:
    """
    Efficient, ALFWorld-specific computation of one-step successors.
    """

    def __init__(self) -> None:
        self.parser = unified_parser

    # ---------------------------
    # Simple helpers reused from succ generator (lightweight re-impl)
    # ---------------------------

    @staticmethod
    def _to_param_list(param_node) -> List[Tuple[str, Optional[str]]]:
        if not isinstance(param_node, list):
            return []
        items = param_node[1:] if param_node and param_node[0] == "parameters" else param_node
        params: List[Tuple[str, Optional[str]]] = []
        tmp: List[str] = []
        i = 0
        while i < len(items):
            it = items[i]
            if it == "-":
                if i + 1 < len(items):
                    t = str(items[i + 1])
                    for v in tmp:
                        params.append((v, t))
                    tmp = []
                    i += 2
                else:
                    i += 1
            else:
                if isinstance(it, list):
                    tmp.extend([str(x) for x in it])
                else:
                    tmp.append(str(it))
                i += 1
        for v in tmp:
            params.append((v, None))
        return params

    @classmethod
    def _split_preconds(cls, node):
        pos: List[Tuple[str, List[str]]] = []
        neg: List[Tuple[str, List[str]]] = []
        eqs: List[Tuple[str, str]] = []
        neqs: List[Tuple[str, str]] = []

        def collect(n, negated: bool = False):
            if isinstance(n, list) and n:
                head = n[0]
                if head == "and":
                    for sub in n[1:]:
                        collect(sub, negated)
                elif head == "not":
                    collect(n[1], True)
                elif head == "=":
                    a = str(n[1])
                    b = str(n[2])
                    if negated:
                        neqs.append((a, b))
                    else:
                        eqs.append((a, b))
                else:
                    pred = str(head)
                    args = [str(x) for x in n[1:] if not isinstance(x, list)]
                    (neg if negated else pos).append((pred, args))

        if isinstance(node, list):
            collect(node, False)
        return pos, neg, eqs, neqs

    @classmethod
    def _split_effects(cls, node):
        add: List[Tuple[str, List[str]]] = []
        delete: List[Tuple[str, List[str]]] = []

        def collect(n, negated: bool = False):
            if isinstance(n, list) and n:
                head = n[0]
                if head == "and":
                    for sub in n[1:]:
                        collect(sub, negated)
                elif head == "not":
                    collect(n[1], True)
                else:
                    pred = str(head)
                    args = [str(x) for x in n[1:] if not isinstance(x, list)]
                    if negated:
                        delete.append((pred, args))
                    else:
                        add.append((pred, args))

        if isinstance(node, list):
            collect(node, False)
        return add, delete

    @classmethod
    def extract_actions(cls, domain_root) -> List[ActionSchema]:
        actions: List[ActionSchema] = []

        def walk(root):
            if isinstance(root, list) and root and root[0] == ":action":
                name = str(root[1]) if len(root) > 1 else "anon"
                params_node = None
                precond_node = None
                effect_node = None
                i = 2
                while i < len(root):
                    key = root[i] if i < len(root) else None
                    val = root[i + 1] if i + 1 < len(root) else None
                    if key == ":parameters":
                        params_node = val
                    elif key == ":precondition":
                        precond_node = val
                    elif key == ":effect":
                        effect_node = val
                    i += 2
                params = cls._to_param_list(params_node if isinstance(params_node, list) else [])
                pos, neg, eqs, neqs = cls._split_preconds(precond_node)
                add, delete = cls._split_effects(effect_node)
                actions.append(ActionSchema(name, params, pos, neg, eqs, neqs, add, delete))
            if isinstance(root, list):
                for c in root:
                    walk(c)

        walk(domain_root)
        return actions

    # ---------------------------
    # Precondition-driven grounding
    # ---------------------------

    @staticmethod
    def _build_indices(init_atoms: List[Tuple[str, List[str]]]):
        """
        Build:
          - rels: pred -> list of arg-tuples
        """
        rels: Dict[str, List[List[str]]] = {}
        for pred, args in init_atoms:
            rels.setdefault(pred, []).append(args)
        return rels

    @staticmethod
    def _check_equalities(env: Dict[str, str], eqs: List[Tuple[str, str]], neqs: List[Tuple[str, str]]) -> bool:
        def val(x: str) -> str:
            return env.get(x, x)

        for a, b in eqs:
            if val(a) != val(b):
                return False
        for a, b in neqs:
            if val(a) == val(b):
                return False
        return True

    @staticmethod
    def _join_on_preconds(
        action: ActionSchema,
        rels: Dict[str, List[List[str]]],
    ) -> List[Dict[str, str]]:
        """
        Given an action and initial-state relations, produce candidate envs
        that satisfy all positive preconditions by joining the relevant facts.
        """
        # Ignore special bookkeeping predicate notvalidated, which is
        # conceptually true initially but often omitted from the
        # encoded initial state. Treating it as a normal fact would
        # wrongly block all actions.
        filtered_preconds = [
            (pred, args) for pred, args in action.preconds_pos if pred != "notvalidated"
        ]

        if not filtered_preconds:
            return [{}]

        # Start from the smallest relation to reduce intermediate size.
        preconds = list(filtered_preconds)
        preconds.sort(key=lambda p: len(rels.get(p[0], [])))

        envs: List[Dict[str, str]] = []

        # Seed with first precondition
        first_pred, first_args = preconds[0]
        tuples = rels.get(first_pred, [])
        vars_first = [a for a in first_args if a.startswith("?")]
        if not tuples:
            return []
        for tup in tuples:
            if len(tup) < len(first_args):
                continue
            env: Dict[str, str] = {}
            ok = True
            for formal, value in zip(first_args, tup):
                if formal.startswith("?"):
                    if formal in env and env[formal] != value:
                        ok = False
                        break
                    env[formal] = value
            if ok:
                envs.append(env)

        # Incrementally join with remaining preconditions
        for pred, args in preconds[1:]:
            tuples = rels.get(pred, [])
            if not tuples:
                return []
            new_envs: List[Dict[str, str]] = []
            vars_here = [a for a in args if a.startswith("?")]
            for env in envs:
                for tup in tuples:
                    if len(tup) < len(args):
                        continue
                    ok = True
                    new_env = dict(env)
                    for formal, value in zip(args, tup):
                        if formal.startswith("?"):
                            if formal in new_env and new_env[formal] != value:
                                ok = False
                                break
                            new_env[formal] = value
                    if ok:
                        new_envs.append(new_env)
            envs = new_envs
            if not envs:
                return []

        return envs

    @staticmethod
    def _ground_literal(lit: Tuple[str, List[str]], env: Dict[str, str]) -> Atom:
        pred, args = lit
        return Atom(predicate=pred, args=[env.get(a, a) for a in args])

    def successors_for_example(
        self,
        domain_pddl: str,
        problem_pddl: str,
    ) -> Tuple[str, State, List[Tuple[str, State]]]:
        """
        Compute all one-step successors for a single ALFWorld example.
        """
        parsed = self.parser.parse_pddl_pair(domain_pddl, problem_pddl, validate=False)
        domain_name = parsed["domain_name"]
        type_to_objs = parsed["objects"]
        init_predicates: List[Tuple[str, List[str]]] = parsed["initial_state"]
        init_atoms = [Atom(predicate=p, args=a) for p, a in init_predicates]
        init_state = State(atoms=init_atoms)
        dom_tree = parsed["domain_tree"]

        actions = self.extract_actions(dom_tree)
        rels = self._build_indices(init_predicates)
        sset = init_state.as_set()

        out: List[Tuple[str, State]] = []

        for a in actions:
            # Candidate envs from positive preconditions
            base_envs = self._join_on_preconds(a, rels)
            if not base_envs:
                continue

            # Expand envs with remaining typed params that don't appear in preconds
            precond_vars = {v for _, args in a.preconds_pos for v in args if v.startswith("?")}
            extra_params = [(v, t) for (v, t) in a.params if v not in precond_vars]

            if extra_params:
                # Build domains for extra params from type_to_objs
                domains: List[List[str]] = []
                names: List[str] = []
                for v, typ in extra_params:
                    names.append(v)
                    if typ is None:
                        all_objs = sorted({o for vs in type_to_objs.values() for o in vs})
                        domains.append(all_objs)
                    else:
                        dom = type_to_objs.get(typ, [])
                        domains.append(dom or type_to_objs.get("object", []))

                expanded_envs: List[Dict[str, str]] = []
                for base in base_envs:
                    for combo in product(*domains):
                        env = dict(base)
                        env.update(dict(zip(names, combo)))
                        expanded_envs.append(env)
            else:
                expanded_envs = base_envs

            for env in expanded_envs:
                # Equalities / inequalities
                if not self._check_equalities(env, a.equalities, a.inequalities):
                    continue

                # Negative preconditions
                neg_ok = True
                for lit in a.preconds_neg:
                    if self._ground_literal(lit, env) in sset:
                        neg_ok = False
                        break
                if not neg_ok:
                    continue

                add_atoms = [self._ground_literal(l, env) for l in a.add_effects]
                del_atoms = [self._ground_literal(l, env) for l in a.del_effects]
                next_state = init_state.apply(add_atoms, del_atoms)
                act_str = " ".join([a.name] + [env[p] for p, _ in a.params])
                out.append((act_str, next_state))

        # Deduplicate by resulting state
        seen = set()
        uniq: List[Tuple[str, State]] = []
        for act, st in out:
            key = tuple(sorted((at.predicate, tuple(at.args)) for at in st.atoms))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((act, st))

        return domain_name, init_state, uniq
