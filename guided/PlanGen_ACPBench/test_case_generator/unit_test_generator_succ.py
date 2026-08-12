#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate initial->next state pairs (one-step successors) for N examples per domain,
optionally restricted to a specific domain name.

- Walks a `train/` folder containing JSON files with keys like "PDDL_domain" and "PDDL_problem".
- Uses unified PDDL parser with tarski support and robust fallback parsing.
- Applies STRIPS semantics: conjunctive preconditions; effects are add/delete literals.

NEW:
    --domain "<name>"  # Only process examples whose (define (domain NAME)) matches/contains this name.

OUTPUT:
    JSON file with test cases, each containing:
      - example_id: unique identifier
      - current_state: list of atoms representing the initial state
      - action: the action applied
      - next_state: list of atoms representing the resulting state

Usage:
    python test_case_generator/unit_test_generator_succ.py --train ./data/train --N 10 --domain ferry --out succ_tests.json
    python unit_test_generator_succ.py --train ./data/train --N 3 --domain blocksworld
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, final
from pydantic import BaseModel, Field, ConfigDict

# Import unified parser
from unified_pddl_parser import unified_parser
from typing import Any

# Define type alias locally to avoid recursive type issues  
Token = Union[str, List[Any]]


# ---------------------------
# Pydantic Atom / State
# ---------------------------

class Atom(BaseModel):
    model_config = ConfigDict(frozen=True)
    predicate: str
    args: List[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.args)))

    def __str__(self) -> str:
        if self.args:
            return f"{self.predicate}({', '.join(self.args)})"
        return self.predicate


class State(BaseModel):
    model_config = ConfigDict(frozen=True)
    atoms: List[Atom] = Field(default_factory=list)

    def as_set(self) -> set:
        return set(self.atoms)

    @classmethod
    def from_iter(cls, atoms: Iterable[Atom]) -> "State":
        return cls(atoms=list({a for a in atoms}))

    def contains(self, atom: Atom) -> bool:
        return atom in self.as_set()

    def apply(self, add: Iterable[Atom], delete: Iterable[Atom]) -> "State":
        s = self.as_set()
        s.difference_update(delete)
        s.update(add)
        return State.from_iter(s)


# ---------------------------
# Simple S-Expression Parser
# ---------------------------

Token = Union[str, List["Token"]]


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


class SuccessorUnitTestGenerator:
    """
    Wraps all previously 'loose' functions into a single class for easier reuse and testing.
    """

    def __init__(self, train_dir: Path, N: int, domain: Optional[str], out: Optional[Path]) -> None:
        self.train_dir = train_dir
        self.N = N
        self.domain = domain
        self.out = out
        # Use unified parser instead of direct tarski
        self.parser = unified_parser
        self.TARKSI_AVAILABLE = unified_parser.tarski_available

    # ---------------------------
    # S-Expression Parser Methods
    # ---------------------------

    @staticmethod
    def tokenize(pddl: str) -> List[str]:
        pddl = re.sub(r";;?.*$", "", pddl, flags=re.MULTILINE)  # strip comments
        pddl = pddl.replace("\t", " ")
        return re.findall(r"\(|\)|[^\s()]+", pddl)

    @classmethod
    def parse_tokens(cls, tokens: List[str], idx: int = 0) -> Tuple[Token, int]:
        lst: List[Token] = []
        assert tokens[idx] == "(", "Expected '('"
        idx += 1
        while idx < len(tokens):
            t = tokens[idx]
            if t == "(":
                node, idx = cls.parse_tokens(tokens, idx)
                lst.append(node)
            elif t == ")":
                return lst, idx + 1
            else:
                lst.append(t)
                idx += 1
        raise ValueError("Unbalanced parentheses")

    @classmethod
    def sexpr(cls, pddl: str) -> Token:
        toks = cls.tokenize(pddl)
        node, i = cls.parse_tokens(toks, 0)
        return node

    # ---------------------------
    # Parsing helpers (fallback)
    # ---------------------------

    @staticmethod
    def find_domain_name(domain_root: Token) -> str:
        # domain root: (define (domain NAME) ...)
        assert isinstance(domain_root, list) and len(domain_root) >= 2
        for i, elt in enumerate(domain_root):
            if isinstance(elt, list) and len(elt) >= 2 and elt[0] == "domain":
                return str(elt[1])
        for elt in domain_root:
            if isinstance(elt, list):
                for sub in elt:
                    if isinstance(sub, list) and len(sub) >= 2 and sub[0] == "domain":
                        return str(sub[1])
        return "unknown-domain"

    @staticmethod
    def extract_objects(problem_root: Token) -> Dict[str, List[str]]:
        type_to_objs: Dict[str, List[str]] = defaultdict(list)
        untyped: List[str] = []

        def walk(root: Token):
            if isinstance(root, list) and len(root) > 0 and root[0] == ":objects":
                items = root[1:]
                tmp: List[str] = []
                i = 0
                while i < len(items):
                    it = items[i]
                    if it == "-":
                        t = str(items[i + 1])
                        for obj in tmp:
                            type_to_objs[t].append(str(obj))
                        tmp = []
                        i += 2
                    else:
                        if isinstance(it, list):
                            tmp.extend([str(x) for x in it])
                        else:
                            tmp.append(str(it))
                        i += 1
                if tmp:
                    untyped.extend([str(x) for x in tmp])

            if isinstance(root, list):
                for c in root:
                    walk(c)
        walk(problem_root)

        if untyped:
            type_to_objs["_any"].extend(untyped)
        return type_to_objs

    @classmethod
    def extract_init_atoms(cls, problem_root: Token) -> List[Atom]:
        init_expr: Optional[Token] = None

        def walk(root: Token):
            nonlocal init_expr
            if isinstance(root, list) and len(root) > 0 and root[0] == ":init":
                init_expr = root
            if isinstance(root, list):
                for c in root:
                    walk(c)
        walk(problem_root)

        if not isinstance(init_expr, list):
            return []

        def literals_from(expr: Token) -> List[Atom]:
            def collect(node: Token) -> List[Atom]:
                if isinstance(node, list) and len(node) > 0:
                    head = node[0]
                    if head == "and":
                        out: List[Atom] = []
                        for sub in node[1:]:
                            out.extend(collect(sub))
                        return out
                    if head == "not":
                        return []
                    pred = str(head)
                    args = [str(x) for x in node[1:] if not isinstance(x, list)]
                    return [Atom(predicate=pred, args=args)]
                return []
            all_atoms: List[Atom] = []
            for sub in expr[1:]:
                all_atoms.extend(collect(sub))
            return all_atoms

        return literals_from(init_expr)

    @staticmethod
    def _to_param_list(param_node: Token) -> List[Tuple[str, Optional[str]]]:
        if not isinstance(param_node, list):
            return []
        items = param_node[1:] if param_node and param_node[0] == "parameters" else param_node
        params: List[Tuple[str, Optional[str]]] = []
        tmp: List[str] = []
        i = 0
        while i < len(items):
            it = items[i]
            if it == "-":
                t = str(items[i + 1])
                for v in tmp:
                    params.append((v, t))
                tmp = []
                i += 2
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
    def _split_preconds(cls, node: Token):
        pos: List[Tuple[str, List[str]]] = []
        neg: List[Tuple[str, List[str]]] = []
        eqs: List[Tuple[str, str]] = []
        neqs: List[Tuple[str, str]] = []

        def collect(n: Token, negated: bool = False):
            if isinstance(n, list) and n:
                head = n[0]
                if head == "and":
                    for sub in n[1:]:
                        collect(sub, negated)
                elif head == "not":
                    collect(n[1], True)
                elif head == "=":
                    a = str(n[1]); b = str(n[2])
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
    def _split_effects(cls, node: Token):
        add: List[Tuple[str, List[str]]] = []
        delete: List[Tuple[str, List[str]]] = []

        def collect(n: Token, negated: bool = False):
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
    def extract_type_hierarchy(cls, domain_root: Token) -> Dict[str, List[str]]:
        """
        Returns mapping child_type -> list of parent types from any :types sections.
        """
        hierarchy: Dict[str, List[str]] = defaultdict(list)

        def record_types(items: Sequence[Token]):
            tmp: List[str] = []
            i = 0
            while i < len(items):
                item = items[i]
                if item == "-":
                    parent = str(items[i + 1]) if i + 1 < len(items) else "object"
                    for child in tmp:
                        hierarchy[child].append(parent)
                    tmp = []
                    i += 2
                else:
                    if isinstance(item, list):
                        tmp.extend(str(x) for x in item)
                    else:
                        tmp.append(str(item))
                    i += 1
            # Types without explicit parent default to object
            for child in tmp:
                hierarchy[child].append("object")

        def walk(node: Token):
            if isinstance(node, list) and node:
                if node[0] == ":types":
                    record_types(node[1:])
                for sub in node:
                    walk(sub)

        walk(domain_root)
        return hierarchy

    @classmethod
    def expand_type_objects(cls, base_objects: Dict[str, List[str]], domain_root: Token) -> Dict[str, List[str]]:
        """
        Augment base objects with entries for every parent type so typed parameters
        (e.g., ?x - locatable) can draw from all subtype objects.
        """
        base: Dict[str, List[str]] = {k: list(v) for k, v in base_objects.items()}
        hierarchy = cls.extract_type_hierarchy(domain_root)
        parent_to_children: Dict[str, List[str]] = defaultdict(list)
        for child, parents in hierarchy.items():
            for parent in parents:
                parent_to_children[parent].append(child)

        all_types = set(base.keys()) | set(hierarchy.keys()) | set(parent_to_children.keys())
        if not all_types:
            return base

        @lru_cache(maxsize=None)
        def gather(typ: str) -> Tuple[str, ...]:
            objs = set(base.get(typ, []))
            for child in parent_to_children.get(typ, []):
                objs.update(gather(child))
            return tuple(sorted(objs))

        expanded: Dict[str, List[str]] = {}
        for typ in sorted(all_types):
            expanded[typ] = list(gather(typ))

        all_objs = sorted({obj for objs in base.values() for obj in objs})
        if all_objs:
            expanded["_any"] = all_objs
        return expanded

    @classmethod
    def extract_actions(cls, domain_root: Token) -> List[ActionSchema]:
        actions: List[ActionSchema] = []

        def walk(root: Token):
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
    # Grounding & applicability
    # ---------------------------

    @staticmethod
    def all_groundings(params: List[Tuple[str, Optional[str]]], type_to_objs: Dict[str, List[str]]) -> Iterable[Dict[str, str]]:
        domains: List[List[str]] = []
        vars_: List[str] = []
        all_objs = sorted({o for objs in type_to_objs.values() for o in objs}) or []
        for var, typ in params:
            vars_.append(var)
            if typ is None:
                domains.append(all_objs)
            else:
                dom = type_to_objs.get(typ, [])
                domains.append(dom or type_to_objs.get("_any", []))
        for tup in product(*domains):
            yield dict(zip(vars_, tup))

    @staticmethod
    def substitute(args: List[str], env: Dict[str, str]) -> List[str]:
        return [env.get(a, a) for a in args]

    @classmethod
    def ground_literal(cls, lit: Tuple[str, List[str]], env: Dict[str, str]) -> Atom:
        pred, args = lit
        return Atom(predicate=pred, args=cls.substitute(args, env))

    @staticmethod
    def check_equalities(env: Dict[str, str], eqs: List[Tuple[str, str]], neqs: List[Tuple[str, str]]) -> bool:
        def val(x: str) -> str:
            return env.get(x, x)
        for a, b in eqs:
            if val(a) != val(b):
                return False
        for a, b in neqs:
            if val(a) == val(b):
                return False
        return True

    def successors_from_init_fallback(self, domain_pddl: str, problem_pddl: str) -> Tuple[str, State, List[Tuple[str, State]]]:
        # Use unified parser for parsing
        try:
            parsed_results = self.parser.parse_pddl_pair(domain_pddl, problem_pddl, validate=False)
            domain_name = parsed_results["domain_name"]
            type_to_objs = parsed_results["objects"]
            
            # Convert unified parser output to our format
            initial_state_predicates = parsed_results["initial_state"]
            init_atoms = [Atom(predicate=pred, args=args) for pred, args in initial_state_predicates]
            init_state = State(atoms=init_atoms)
            
            dom_tree = parsed_results["domain_tree"]
            type_to_objs = self.expand_type_objects(type_to_objs, dom_tree)
        except Exception as e:
            print(f"Warning: Unified parser failed, using fallback S-expression parsing: {e}")
            # Fallback to original method
            dom = self.sexpr(domain_pddl)
            prob = self.sexpr(problem_pddl)
            domain_name = self.find_domain_name(dom)
            type_to_objs = self.extract_objects(prob)
            init_atoms = self.extract_init_atoms(prob)
            init_state = State(atoms=init_atoms)
            dom_tree = dom
            type_to_objs = self.expand_type_objects(type_to_objs, dom_tree)

        actions = self.extract_actions(dom_tree)
        out: List[Tuple[str, State]] = []

        sset = init_state.as_set()

        for a in actions:
            for env in self.all_groundings(a.params, type_to_objs):
                if not self.check_equalities(env, a.equalities, a.inequalities):
                    continue
                pos_ok = all(self.ground_literal(l, env) in sset for l in a.preconds_pos)
                neg_ok = all(self.ground_literal(l, env) not in sset for l in a.preconds_neg)
                if not (pos_ok and neg_ok):
                    continue
                add_atoms = [self.ground_literal(l, env) for l in a.add_effects]
                del_atoms = [self.ground_literal(l, env) for l in a.del_effects]
                next_state = init_state.apply(add_atoms, del_atoms)
                act_str = " ".join([a.name] + [env[p] for p, _ in a.params])
                out.append((act_str, next_state))

        seen = set()
        uniq: List[Tuple[str, State]] = []
        for act, st in out:
            key = tuple(sorted((at.predicate, tuple(at.args)) for at in st.atoms))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((act, st))

        return domain_name, init_state, uniq

    def successors_from_init_tarski(self, domain_pddl: str, problem_pddl: str) -> Tuple[str, State, List[Tuple[str, State]]]:
        # Use unified parser for validation and parsing
        if not self.TARKSI_AVAILABLE:
            return self.successors_from_init_fallback(domain_pddl, problem_pddl)

        # Validate with tarski if available
        if not self.parser.validate_with_tarski(domain_pddl, problem_pddl):
            print("Warning: PDDL validation failed, falling back to S-expression parser")
            return self.successors_from_init_fallback(domain_pddl, problem_pddl)
        
        # Use fallback parser for actual processing (tarski used only for validation)
        return self.successors_from_init_fallback(domain_pddl, problem_pddl)

    # ---------------------------
    # Dataset traversal / filtering
    # ---------------------------

    def iter_examples(self, train_dir: Path, domain_filter: Optional[str] = None) -> Iterable[Tuple[str, int, Dict[str, str]]]:
        for path in sorted(train_dir.rglob("*.json")):
            # Optimization: Filter by filename if domain filter is provided
            if domain_filter:
                # normalize both to allow flexible matching (e.g. frogs_jumping vs frogs-jumping)
                fname = path.name.lower()
                dfilter = domain_filter.lower().replace("_", "").replace("-", "")
                fsimple = fname.replace("_", "").replace("-", "")
                # Simple check: if the filter (without separators) isn't in the filename (without separators), skip
                # This is a heuristic; consistent parsing will confirm later.
                if dfilter not in fsimple:
                    continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                records = data.get("data") or data.get("examples") or data.get("records") or []
            else:
                records = data

            if not isinstance(records, list):
                continue

            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                domain_text = rec.get("PDDL_domain") or rec.get("PDDL_Domain") or rec.get("domain") or rec.get("pddl_domain")
                problem_text = rec.get("PDDL_problem") or rec.get("PDDL_Problem") or rec.get("problem") or rec.get("pddl_problem")
                ex_id = rec.get("id") or rec.get("example_id") or f"{path.name}:{idx}"
                if isinstance(domain_text, str) and isinstance(problem_text, str):
                    yield (str(path), idx, {"id": ex_id, "domain": domain_text, "problem": problem_text})

    def collect_N_per_domain(self) -> Dict[str, List[Tuple[str, str, str]]]:
        all_examples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # Pass domain filter to iterator for speedup
        for file_path, idx, rec in self.iter_examples(self.train_dir, self.domain):
            domain_pddl = rec["domain"]
            problem_pddl = rec["problem"]
            try:
                # Use unified parser for consistent domain name extraction
                domain_name, _ = self.parser.parse_pddl_domain(domain_pddl)
            except Exception as e:
                print(f"Warning: Failed to parse domain in {file_path}:{idx}: {e}")
                domain_name = "unknown-domain"

            # Use unified parser's domain matching
            if not self.parser.domain_matches(domain_name, self.domain):
                continue

            all_examples[domain_name].append((rec["id"], domain_pddl, problem_pddl))

        per_domain: Dict[str, List[Tuple[str, str, str]]] = {}
        for dom_name, examples in all_examples.items():
            if len(examples) > self.N:
                # Optimization: Take FIRST N examples instead of random.
                # Files are typically sorted by complexity; random sampling can pick huge instances
                # that cause grounding explosion.
                per_domain[dom_name] = examples[:self.N]
            else:
                per_domain[dom_name] = examples
        return per_domain

    # ---------------------------
    # Orchestration
    # ---------------------------

    def compute_pairs_for_domain(self, examples: List[Tuple[str, str, str]]) -> List[Dict]:
        results: List[Dict] = []
        for ex_id, dom_txt, prob_txt in examples:
            try:
                # Special optimized path for ALFWorld / Alfred domain
                # to avoid combinatorial blow-up in grounding.
                parsed = self.parser.parse_pddl_pair(dom_txt, prob_txt, validate=False)
                dom_name = parsed["domain_name"]

                if dom_name.lower() in {"alfworld", "alfred"}:
                    try:
                        from .alfworld_succ_generator import AlfworldSuccessorGenerator  # type: ignore
                    except ImportError:
                        from alfworld_succ_generator import AlfworldSuccessorGenerator  # type: ignore

                    alf_gen = AlfworldSuccessorGenerator()
                    dom_name, init_state, succ = alf_gen.successors_for_example(
                        dom_txt, prob_txt
                    )
                else:
                    if self.TARKSI_AVAILABLE:
                        dom_name, init_state, succ = self.successors_from_init_tarski(
                            dom_txt, prob_txt
                        )
                    else:
                        dom_name, init_state, succ = self.successors_from_init_fallback(
                            dom_txt, prob_txt
                        )
            except Exception as e:
                results.append({"example_id": ex_id, "error": f"{type(e).__name__}: {e}"})
                continue

            results.append({
                "example_id": ex_id,
                "initial_state": [a.model_dump() for a in sorted(init_state.as_set(), key=lambda x: (x.predicate, x.args))],
                "next_states": [
                    {"action": act, "state": [a.model_dump() for a in sorted(st.as_set(), key=lambda x: (x.predicate, x.args))]}
                    for (act, st) in succ
                ],
            })
        return results

    def generate_succ_unit_tests(self):
        per_domain = self.collect_N_per_domain()
        if self.domain and not per_domain:
            print(json.dumps({"warning": f"No examples found for domain filter '{self.domain}'"}))
        final: Dict[str, List[Dict]] = {}
        for dom_name, examples in per_domain.items():
            print(f"[SuccGen] Domain '{dom_name}': {len(examples)} examples selected.")
            results = self.compute_pairs_for_domain(examples)
            print(f"[SuccGen] Domain '{dom_name}': finished computing {len(results)} test cases.")
            final[dom_name] = results
        out_text = json.dumps(final, indent=2)
        if self.out is not None:
            self.out.write_text(out_text, encoding="utf-8")
            print(f"Wrote results to {self.out}")
        else:
            print(f"Results:\n\n{out_text}")
        return final, out_text


#################################################################
"""
Example usage to test the successor unit test generator.
"""
#################################################################


def main():
    ap = argparse.ArgumentParser(description="Generate initial->next state pairs from PDDL JSON examples.")
    ap.add_argument("--train", "-t", type=str, required=True, help="Path to the train/ folder")
    ap.add_argument("--N", "-n", type=int, default=1, help="Number of examples per domain")
    ap.add_argument("--domain", "-d", type=str, default=None, help="Restrict to this domain name (substring match), If None process all")
    ap.add_argument("--out", "-o", type=str, default=None, help="Optional path to write JSON output")
    ap.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    args = ap.parse_args()
    random.seed(args.seed)
    train_dir = Path(args.train)
    if not train_dir.exists():
        print(json.dumps({"error": f"Train folder not found: {train_dir}"}), file=sys.stderr)
        sys.exit(2)

    generator = SuccessorUnitTestGenerator(train_dir, args.N, args.domain, Path(args.out))
    final, out_text = generator.generate_succ_unit_tests()
    print(f"Generated {sum(len(v) for v in final.values())} test cases across {len(final)} domains.")

if __name__ == "__main__":
    main()
