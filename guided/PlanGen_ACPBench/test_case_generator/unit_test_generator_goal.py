#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate goal-checker test cases for N examples of a (possibly filtered) domain.

For each example, this produces two tests:
  - True Negative:  current = init,  goal = goal, expected = False
  - True Positive:  current = goal,  goal = goal, expected = True

Output JSON (per domain):
{
  "<domain>": [
    {"example_id": "ex1:init-vs-goal", "current_state": [...], "goal_state": [...], "expected": false},
    {"example_id": "ex1:goal-vs-goal", "current_state": [...], "goal_state": [...], "expected": true},
    ...
  ]
}

Usage:
    python test_case_generator/unit_test_generator_goal.py --train ./data/train --N 10 --domain ferry --out goal_tests.json
    python test_case_generator/unit_test_generator_goal.py --train ./data/train --N 3 --domain ferry
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict

# Import unified parser
from unified_pddl_parser import unified_parser

# ---------------------------
# Optional Tarski import
# ---------------------------
try:
    from tarski.io import PDDLReader  # type: ignore
    TARKSI_AVAILABLE = True
except Exception:
    PDDLReader = None  # type: ignore
    TARKSI_AVAILABLE = False


# ---------------------------
# Pydantic Atom / State
# ---------------------------

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


# ---------------------------
# Simple S-Expression Parser
# ---------------------------

Token = Union[str, List["Token"]]


class GoalTestCaseGenerator:
    """
    Wraps the loose functions into a single reusable class.
    """

    def __init__(self, train_dir: Path, N: int, domain: Optional[str], out: Optional[Path]) -> None:
        self.train_dir = train_dir
        self.N = N
        self.domain = domain
        self.out = out
        # Detect Tarski at runtime
        try:
            from tarski.io import PDDLReader as _PDDLReader  # type: ignore
            self.PDDLReader = _PDDLReader
            self.TARKSI_AVAILABLE = True
        except Exception:
            self.PDDLReader = None  # type: ignore
            self.TARKSI_AVAILABLE = False

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
        node, _ = cls.parse_tokens(toks, 0)
        return node

    # ---------------------------
    # Parsing helpers
    # ---------------------------

    @staticmethod
    def find_domain_name(domain_root: Token) -> str:
        assert isinstance(domain_root, list) and len(domain_root) >= 2
        for elt in domain_root:
            if isinstance(elt, list) and len(elt) >= 2 and elt[0] == "domain":
                return str(elt[1])
            if isinstance(elt, list):
                for sub in elt:
                    if isinstance(sub, list) and len(sub) >= 2 and sub[0] == "domain":
                        return str(sub[1])
        return "unknown-domain"

    @classmethod
    def extract_init_atoms(cls, problem_root: Token) -> List[Atom]:
        init_expr: Optional[Token] = None

        def walk(root: Token):
            nonlocal init_expr
            if isinstance(root, list) and root and root[0] == ":init":
                init_expr = root
            if isinstance(root, list):
                for c in root:
                    walk(c)
        walk(problem_root)

        if not isinstance(init_expr, list):
            return []

        def collect(node: Token) -> List[Atom]:
            if isinstance(node, list) and node:
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

        atoms: List[Atom] = []
        for sub in init_expr[1:]:
            atoms.extend(collect(sub))
        return atoms

    @classmethod
    def extract_goal_atoms(cls, problem_root: Token) -> List[Atom]:
        goal_expr: Optional[Token] = None

        def walk(root: Token):
            nonlocal goal_expr
            if isinstance(root, list) and root and root[0] == ":goal":
                goal_expr = root
            if isinstance(root, list):
                for c in root:
                    walk(c)
        walk(problem_root)

        if not isinstance(goal_expr, list):
            return []

        def collect(node: Token, negated: bool = False) -> List[Atom]:
            if isinstance(node, list) and node:
                head = node[0]
                if head == "and":
                    out: List[Atom] = []
                    for sub in node[1:]:
                        out.extend(collect(sub, negated))
                    return out
                if head == "not":
                    # ignore negated goals for subset semantics
                    return []
                pred = str(head)
                args = [str(x) for x in node[1:] if not isinstance(x, list)]
                return [Atom(predicate=pred, args=args)]
            return []

        atoms: List[Atom] = []
        for sub in goal_expr[1:]:
            atoms.extend(collect(sub))
        return atoms

    # ---------------------------
    # Dataset traversal / filtering
    # ---------------------------

    @staticmethod
    def iter_examples(train_dir: Path, domain_filter: Optional[str] = None):
        for path in sorted(Path(train_dir).rglob("*.json")):
            if domain_filter:
                fname = path.name.lower()
                dfilter = domain_filter.lower().replace("_", "").replace("-", "")
                fsimple = fname.replace("_", "").replace("-", "")
                if dfilter not in fsimple:
                    continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            records = data if isinstance(data, list) else (data.get("data") or data.get("examples") or data.get("records") or [])
            if not isinstance(records, list):
                continue
            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                domain_pddl = rec.get("PDDL_domain") or rec.get("PDDL_Domain") or rec.get("domain") or rec.get("pddl_domain")
                problem_pddl = rec.get("PDDL_problem") or rec.get("PDDL_Problem") or rec.get("problem") or rec.get("pddl_problem")
                ex_id = rec.get("id") or rec.get("example_id") or f"{path.name}:{idx}"
                if isinstance(domain_pddl, str) and isinstance(problem_pddl, str):
                    yield (str(path), idx, {"id": ex_id, "domain": domain_pddl, "problem": problem_pddl})

    @staticmethod
    def domain_matches(name: str, filt: Optional[str]) -> bool:
        if not filt:
            return True
        return filt.lower() in (name or "").lower()

    def collect_N_per_domain(self, train_dir: Path, N: int, domain_filter: Optional[str]):
        all_examples = defaultdict(list)
        for _, _, rec in self.iter_examples(train_dir, domain_filter):
            domain_pddl = rec["domain"]
            try:
                # Use unified parser for consistent domain name extraction
                domain_name, _ = unified_parser.parse_pddl_domain(domain_pddl)
            except Exception:
                domain_name = "unknown-domain"
            
            # Use unified parser's domain matching
            if not unified_parser.domain_matches(domain_name, domain_filter):
                continue
            all_examples[domain_name].append(rec)

        per_domain = {}
        for dom_name, examples in all_examples.items():
            if len(examples) > N:
                per_domain[dom_name] = examples[:N]
            else:
                per_domain[dom_name] = examples
        return per_domain

    # ---------------------------
    # Validation and test creation
    # ---------------------------

    def validate_with_tarski(self, domain_pddl: str, problem_pddl: str) -> None:
        if not self.TARKSI_AVAILABLE:
            return
        with tempfile.TemporaryDirectory() as td:
            dp = Path(td) / "d.pddl"
            pp = Path(td) / "p.pddl"
            dp.write_text(domain_pddl, encoding="utf-8")
            pp.write_text(problem_pddl, encoding="utf-8")
            reader = self.PDDLReader(raise_on_error=True, strict_with_requirements=False)  # type: ignore
            reader.parse_domain(str(dp))
            reader.parse_instance(str(pp))

    def make_tests_for_example(self, ex: Dict[str, str]):
        domain_pddl = ex["domain"]
        problem_pddl = ex["problem"]
        try:
            # Use unified parser validation if available
            if unified_parser.tarski_available:
                unified_parser.validate_with_tarski(domain_pddl, problem_pddl)

            # Parse with unified parser
            parsed_results = unified_parser.parse_pddl_pair(domain_pddl, problem_pddl, validate=False)
            
            # Extract initial and goal states
            initial_state_predicates = parsed_results["initial_state"]
            init_atoms = [Atom(predicate=pred, args=args) for pred, args in initial_state_predicates]
            init_state = State(atoms=init_atoms)
            
            # Extract goal state from problem tree  
            problem_tree = parsed_results["problem_tree"]
            goal_atoms = self.extract_goal_atoms(problem_tree)
            goal_state = State(atoms=goal_atoms)

        except Exception:
            # Fallback to original parsing
            try:
                self.validate_with_tarski(domain_pddl, problem_pddl)
            except Exception:
                pass

            prob_root = self.sexpr(problem_pddl)
            init_state = State(atoms=self.extract_init_atoms(prob_root))
            goal_state = State(atoms=self.extract_goal_atoms(prob_root))

        return [
            {
                "example_id": f"{ex['id']}:init-vs-goal",
                "current_state": [a.model_dump() for a in sorted(init_state.as_set(), key=lambda x: (x.predicate, x.args))],
                "goal_state": [a.model_dump() for a in sorted(goal_state.as_set(), key=lambda x: (x.predicate, x.args))],
                "expected": False,
            },
            {
                "example_id": f"{ex['id']}:goal-vs-goal",
                "current_state": [a.model_dump() for a in sorted(goal_state.as_set(), key=lambda x: (x.predicate, x.args))],
                "goal_state": [a.model_dump() for a in sorted(goal_state.as_set(), key=lambda x: (x.predicate, x.args))],
                "expected": True,
            },
        ]

    def generate_goal_unit_tests(self) -> Dict[str, List[Dict]]:
        per_domain = self.collect_N_per_domain(self.train_dir, self.N, self.domain)
        final: Dict[str, List[Dict]] = {}
        for dom_name, examples in per_domain.items():
            print(f"[GoalGen] Domain '{dom_name}': {len(examples)} examples selected.")
            cases: List[Dict] = []
            for idx, ex in enumerate(examples, start=1):
                cases.extend(self.make_tests_for_example(ex))
                # Lightweight progress indicator every 5 examples
                if idx % 5 == 0 or idx == len(examples):
                    print(f"[GoalGen] Domain '{dom_name}': processed {idx}/{len(examples)} examples.")
            final[dom_name] = cases
        out_text = json.dumps(final, indent=2)
        if self.out is not None:
            self.out.write_text(out_text, encoding="utf-8")
            print(f"Wrote results to {self.out}")
        else:
            print(f"Results:\n\n{out_text}")
        return final, out_text

#################################################################
"""
Example usage to test the goal unit test generator.
"""
#################################################################

def main():
    ap = argparse.ArgumentParser(description="Generate goal-checker test cases from PDDL JSON examples.")
    ap.add_argument("--train", "-t", type=str, required=True, help="Path to the train/ folder")
    ap.add_argument("--N", "-n", type=int, default=10, help="Number of examples per domain")
    ap.add_argument("--domain", "-d", type=str, default=None, help="Restrict to this domain name (substring match) if None process all")
    ap.add_argument("--out", "-o", type=str, default=None, help="Optional path to write JSON output")
    ap.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    args = ap.parse_args()
    random.seed(args.seed)
    train_dir = Path(args.train)
    if not train_dir.exists():
        print(json.dumps({"error": f"Train folder not found: {train_dir}"}), file=sys.stderr)
        sys.exit(2)

    generator = GoalTestCaseGenerator(train_dir, args.N, args.domain, Path(args.out))
    final, out_text = generator.generate_goal_unit_tests()


if __name__ == "__main__":
    main()
