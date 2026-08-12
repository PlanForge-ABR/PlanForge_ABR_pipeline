#!/usr/bin/env python3
"""
Action Validator using KCL-Planning/VAL.
Replaces action_executor.py by running the Validate C++ binary on dynamically generated PDDL.
"""

import os
import glob
import json
import re
import subprocess
import tempfile
import collections
from typing import List, Dict, Tuple, Set, Optional, Any

# Ensure we run from the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
while project_root and not os.path.exists(os.path.join(project_root, "data")):
    parent = os.path.dirname(project_root)
    if parent == project_root:
        break
    project_root = parent
os.chdir(project_root)

# ============================================================================
# HARDCODED PDDL DOMAINS (for domains not present in data/train)
# ============================================================================

HANOI_DOMAIN = """(define (domain hanoi)
  (:requirements :strips)
  (:predicates (on ?x ?y)
               (clear ?x)
               (smaller ?x ?y))
  (:action move
     :parameters (?disk ?from ?to)
     :precondition (and (smaller ?to ?disk) (on ?disk ?from) (clear ?disk) (clear ?to))
     :effect (and (clear ?from) (on ?disk ?to) (not (on ?disk ?from)) (not (clear ?to)))
  )
)"""

FROGS_DOMAIN = """(define (domain frogs-jumping)
  (:requirements :strips)
  (:predicates (at ?frog ?pos)
               (empty ?pos)
               (next ?pos1 ?pos2))
  (:action slide-right
     :parameters (?frog ?from ?to)
     :precondition (and (at ?frog ?from) (empty ?to) (next ?from ?to))
     :effect (and (not (at ?frog ?from)) (at ?frog ?to) (empty ?from) (not (empty ?to)))
  )
  (:action slide-left
     :parameters (?frog ?from ?to)
     :precondition (and (at ?frog ?from) (empty ?to) (next ?to ?from))
     :effect (and (not (at ?frog ?from)) (at ?frog ?to) (empty ?from) (not (empty ?to)))
  )
  (:action jump-right
     :parameters (?lf ?from ?mid ?to ?rf)
     :precondition (and (at ?lf ?from) (at ?rf ?mid) (empty ?to) (next ?from ?mid) (next ?mid ?to))
     :effect (and (not (at ?lf ?from)) (at ?lf ?to) (empty ?from) (not (empty ?to)))
  )
  (:action jump-left
     :parameters (?rf ?from ?mid ?to ?lf)
     :precondition (and (at ?rf ?from) (at ?lf ?mid) (empty ?to) (next ?mid ?from) (next ?to ?mid))
     :effect (and (not (at ?rf ?from)) (at ?rf ?to) (empty ?from) (not (empty ?to)))
  )
)"""

def get_pddl_domain(domain_name: str) -> str:
    domain_lower = domain_name.lower().replace("_", "-")

    # Search in training files and other data directories
    base_names = [domain_name, domain_name.replace("_", "-"), domain_name.replace("-", "_")]
    search_dirs = [
        "data/train",
        "data/test_baseline",
        "data/test_new",
        "data/test_old",
        "data/acp_bench_new"
    ]
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for base in set(base_names):
            pattern = os.path.join(sdir, f"*{base}*.json")
            files = glob.glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            pddl = data[0].get("PDDL_domain")
                            if pddl:
                                return pddl
                except Exception:
                    pass

    # Fallback to hardcoded definitions if not found in data
    if domain_lower == "hanoi":
        return HANOI_DOMAIN
    if domain_lower in ("frogs-jumping", "frogs_jumping"):
        return FROGS_DOMAIN

    raise ValueError(f"Could not find PDDL domain for {domain_name}")

# ============================================================================
# PDDL PARSING UTILITIES
# ============================================================================

def parse_facts_syntax(state_str: str) -> List[Tuple[str, List[str]]]:
    """Parse PDDL-style fact strings into (predicate, args) tuples."""
    if not state_str or not isinstance(state_str, str):
        return []

    def parse_expr(expr: str) -> List[Tuple[str, List[str]]]:
        expr = expr.strip()
        if not expr:
            return []
        while expr.startswith("((") and expr.endswith("))"):
            expr = expr[1:-1].strip()
        if expr.startswith("(") and expr.endswith(")"):
            expr_inner = expr[1:-1].strip()
        else:
            expr_inner = expr
        if not expr_inner:
            return []
        if expr_inner.lower().startswith("and "):
            inner = expr_inner[3:].strip()
            results: List[Tuple[str, List[str]]] = []
            balance = 0
            current: List[str] = []
            for ch in inner:
                if ch == "(":
                    balance += 1
                elif ch == ")":
                    balance -= 1
                current.append(ch)
                if balance == 0 and ch == ")" and "".join(current).strip():
                    results.extend(parse_expr("".join(current).strip()))
                    current = []
            return results
        if expr_inner.lower().startswith("not "):
            return []
        parts = expr_inner.split()
        if not parts:
            return []
        return [(parts[0], parts[1:])]

    facts: List[Tuple[str, List[str]]] = []
    i = 0
    n = len(state_str)
    while i < n:
        if state_str[i] == "(":
            start = i
            depth = 1
            i += 1
            while i < n and depth > 0:
                if state_str[i] == "(":
                    depth += 1
                elif state_str[i] == ")":
                    depth -= 1
                i += 1
            facts.extend(parse_expr(state_str[start:i]))
        else:
            i += 1
    return facts

def parse_actions_syntax(action_str: str) -> List[Tuple[str, List[str]]]:
    """Parse action sequences from LLM output."""
    if not action_str or not isinstance(action_str, str):
        return []

    def parse_expr(expr: str) -> List[Tuple[str, List[str]]]:
        expr = expr.strip()
        if not expr:
            return []
        while expr.startswith("((") and expr.endswith("))"):
            expr = expr[1:-1].strip()
        if expr.startswith("(") and expr.endswith(")"):
            inner = expr[1:-1].strip()
        else:
            inner = expr
        if not inner:
            return []
        if inner.lower().startswith("and "):
            body = inner[3:].strip()
            results: List[Tuple[str, List[str]]] = []
            balance = 0
            current: List[str] = []
            for ch in body:
                if ch == "(":
                    balance += 1
                elif ch == ")":
                    balance -= 1
                current.append(ch)
                if balance == 0 and ch == ")" and "".join(current).strip():
                    results.extend(parse_expr("".join(current).strip()))
                    current = []
            return results
        if inner.lower().startswith("not "):
            return []
        parts = inner.split()
        if not parts:
            return []
        return [(parts[0], parts[1:])]

    actions: List[Tuple[str, List[str]]] = []
    found_parens = False
    i = 0
    n = len(action_str)
    while i < n:
        if action_str[i] == "(":
            found_parens = True
            start = i
            depth = 1
            i += 1
            while i < n and depth > 0:
                if action_str[i] == "(":
                    depth += 1
                elif action_str[i] == ")":
                    depth -= 1
                i += 1
            actions.extend(parse_expr(action_str[start:i]))
        else:
            i += 1

    if found_parens and actions:
        return actions

    # Fallback: line-based parsing
    for line in action_str.split("\n"):
        line = line.strip()
        line = re.sub(r"^\d+[\.\\)]\s*", "", line)
        line = re.sub(r"^-\s*", "", line)
        if not line:
            continue
        line = line.replace('"', "").replace("'", "").replace("[", "").replace("]", "").replace(",", "")
        parts = line.split()
        if parts:
            actions.append((parts[0], parts[1:]))
    return actions

def parse_action_string(action_str: str) -> Optional[Tuple[str, List[str]]]:
    """Helper to parse action(args) or (action args)."""
    if not action_str:
        return None
    action_str = action_str.strip()
    # name(arg1, arg2)
    match = re.match(r"(\w[\w\-]*)\((.*)\)", action_str)
    if match:
        name = match.group(1)
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        return name, args
    # (name arg1 arg2)
    if action_str.startswith("(") and action_str.endswith(")"):
        parts = action_str[1:-1].split()
        if parts:
            return parts[0], parts[1:]
    # Simple name arg1 arg2
    parts = action_str.split()
    if parts:
        return parts[0], parts[1:]
    return None

# ============================================================================
# TYPE INFERENCE & PROBLEM GENERATION
# ============================================================================

def get_inside_parens(text: str, keyword: str) -> str:
    # Normalize whitespace
    text_norm = re.sub(r'\s+', ' ', text)
    idx = text_norm.lower().find(keyword.lower())
    if idx == -1:
        return ""
    start_idx = text_norm.find("(", idx)
    if start_idx == -1:
        return ""
    count = 1
    for i in range(start_idx + 1, len(text_norm)):
        if text_norm[i] == '(':
            count += 1
        elif text_norm[i] == ')':
            count -= 1
        if count == 0:
            return text_norm[start_idx:i+1]
    return ""

def parse_predicates_signature(pddl_domain: str) -> Dict[str, List[str]]:
    predicates_block = get_inside_parens(pddl_domain, "(:predicates")
    if not predicates_block:
        return {}
    
    inner_preds = []
    idx = predicates_block.find('(')
    scan_idx = idx + 1
    while scan_idx < len(predicates_block) - 1:
        start = predicates_block.find('(', scan_idx)
        if start == -1 or start == len(predicates_block) - 1:
            break
        count = 1
        for i in range(start + 1, len(predicates_block)):
            if predicates_block[i] == '(':
                count += 1
            elif predicates_block[i] == ')':
                count -= 1
            if count == 0:
                inner_preds.append(predicates_block[start+1:i].strip())
                scan_idx = i + 1
                break
        else:
            break

    pred_types = {}
    for pred_str in inner_preds:
        tokens = pred_str.split()
        if not tokens:
            continue
        pred_name = tokens[0].lower()
        param_vars = []
        vars_accum = []
        types_dict = {}
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith('?'):
                vars_accum.append(tok)
                param_vars.append(tok)
                i += 1
            elif tok == '-':
                if i + 1 < len(tokens):
                    t = tokens[i+1]
                    for v in vars_accum:
                        types_dict[v] = t.lower()
                    vars_accum = []
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        for v in vars_accum:
            types_dict[v] = 'object'
        
        pred_types[pred_name] = [types_dict[v] for v in param_vars]
    return pred_types

def parse_types_hierarchy(pddl_domain: str) -> Dict[str, str]:
    types_block = get_inside_parens(pddl_domain, "(:types")
    if not types_block:
        return {}
    
    tokens = types_block.strip().replace(')', '').split()
    if not tokens:
        return {}
    
    if tokens[0].lower() == ':types':
        tokens = tokens[1:]
        
    parent_map = {}
    children_accum = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == '-':
            if i + 1 < len(tokens):
                parent = tokens[i+1].lower()
                for c in children_accum:
                    parent_map[c.lower()] = parent
                children_accum = []
                i += 2
            else:
                i += 1
        else:
            children_accum.append(tok)
            i += 1
    for c in children_accum:
        c_lower = c.lower()
        if c_lower != 'object':
            parent_map[c_lower] = 'object'
    return parent_map

def infer_object_types(
    init_facts: List[Tuple[str, List[str]]],
    goal_facts: List[Tuple[str, List[str]]],
    pred_types: Dict[str, List[str]],
    parent_map: Dict[str, str]
) -> Dict[str, str]:
    obj_candidate_types = collections.defaultdict(set)
    all_objs = set()
    for pred, args in init_facts + goal_facts:
        for arg in args:
            all_objs.add(arg)
            
    for pred, args in init_facts + goal_facts:
        pred_lower = pred.lower()
        if pred_lower in pred_types:
            expected_types = pred_types[pred_lower]
            for j, arg in enumerate(args):
                if j < len(expected_types):
                    obj_candidate_types[arg].add(expected_types[j])

    def is_subtype(t1: str, t2: str) -> bool:
        if t1 == t2:
            return True
        curr = t1
        visited = set()
        while curr in parent_map:
            curr = parent_map[curr]
            if curr in visited:
                break
            visited.add(curr)
            if curr == t2:
                return True
        return False

    def get_more_specific_type(t1: str, t2: str) -> str:
        if is_subtype(t1, t2):
            return t1
        if is_subtype(t2, t1):
            return t2
        return t1

    obj_types = {}
    for obj in all_objs:
        candidates = obj_candidate_types.get(obj, set())
        if not candidates:
            obj_types[obj] = 'object'
            continue
        
        resolved = None
        for cand in candidates:
            if resolved is None:
                resolved = cand
            else:
                resolved = get_more_specific_type(resolved, cand)
        obj_types[obj] = resolved
    return obj_types

def get_domain_name(pddl_domain: str) -> str:
    match = re.search(r'\(domain\s+([^\s()]+)\)', pddl_domain, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown-domain"

def generate_problem_pddl(
    domain_name: str,
    example_id: str,
    init_facts: List[Tuple[str, List[str]]],
    goal_facts: List[Tuple[str, List[str]]],
    obj_types: Dict[str, str],
    use_typing: bool
) -> str:
    problem_name = example_id.replace(":", "-").replace(".", "-").replace("/", "-")
    
    if use_typing:
        type_to_objs = collections.defaultdict(list)
        for obj, t in obj_types.items():
            type_to_objs[t].append(obj)
        objects_lines = []
        for t in sorted(type_to_objs.keys()):
            objs = sorted(type_to_objs[t])
            objects_lines.append(f"        {' '.join(objs)} - {t}")
        objects_str = "\n".join(objects_lines)
    else:
        objects_str = f"        {' '.join(sorted(obj_types.keys()))}"

    init_lines = []
    for pred, args in init_facts:
        if args:
            init_lines.append(f"        ({pred} {' '.join(args)})")
        else:
            init_lines.append(f"        ({pred})")
    init_str = "\n".join(init_lines)

    goal_lines = []
    for pred, args in goal_facts:
        if args:
            goal_lines.append(f"        ({pred} {' '.join(args)})")
        else:
            goal_lines.append(f"        ({pred})")
    
    if not goal_lines:
        goal_str = "(and)"
    elif len(goal_lines) == 1:
        goal_str = goal_lines[0].strip()
    else:
        goal_str = f"(and\n{chr(10).join(goal_lines)}\n      )"

    pddl = f"""(define (problem {problem_name})
    (:domain {domain_name})
    (:objects
{objects_str}
    )
    (:init
{init_str}
    )
    (:goal
        {goal_str}
    )
)"""
    return pddl

# ============================================================================
# VERIFICATION ENGINE
# ============================================================================

def fix_pddl_domain_typing(pddl_domain: str) -> str:
    if ":typing" not in pddl_domain.lower():
        return pddl_domain
    
    predicates_block = get_inside_parens(pddl_domain, "(:predicates")
    if not predicates_block:
        return pddl_domain

    idx = predicates_block.find('(')
    scan_idx = idx + 1
    new_preds = []
    
    while scan_idx < len(predicates_block) - 1:
        start = predicates_block.find('(', scan_idx)
        if start == -1 or start == len(predicates_block) - 1:
            break
        count = 1
        for i in range(start + 1, len(predicates_block)):
            if predicates_block[i] == '(':
                count += 1
            elif predicates_block[i] == ')':
                count -= 1
            if count == 0:
                pred_str = predicates_block[start+1:i].strip()
                tokens = pred_str.split()
                if tokens:
                    pred_name = tokens[0]
                    vars_accum = []
                    typed_tokens = [pred_name]
                    j = 1
                    while j < len(tokens):
                        tok = tokens[j]
                        if tok.startswith('?'):
                            vars_accum.append(tok)
                            j += 1
                        elif tok == '-':
                            if j + 1 < len(tokens):
                                t = tokens[j+1]
                                for v in vars_accum:
                                    typed_tokens.append(f"{v} - {t}")
                                vars_accum = []
                                j += 2
                            else:
                                j += 1
                        else:
                            j += 1
                    for v in vars_accum:
                        typed_tokens.append(f"{v} - object")
                    
                    new_pred_str = f"({ ' '.join(typed_tokens) })"
                    new_preds.append(new_pred_str)
                scan_idx = i + 1
                break
        else:
            break

    new_predicates_block = f"(:predicates\n  {chr(10).join(new_preds)}\n)"
    
    idx_orig = pddl_domain.lower().find("(:predicates")
    if idx_orig != -1:
        count = 1
        start_idx = pddl_domain.find("(", idx_orig)
        if start_idx != -1:
            for i in range(start_idx + 1, len(pddl_domain)):
                if pddl_domain[i] == '(':
                    count += 1
                elif pddl_domain[i] == ')':
                    count -= 1
                if count == 0:
                    orig_block = pddl_domain[idx_orig:i+1]
                    pddl_domain = pddl_domain.replace(orig_block, new_predicates_block)
                    break
    return pddl_domain

def refine_types_by_name(obj_types: Dict[str, str], parent_map: Dict[str, str]) -> Dict[str, str]:
    all_types = set(parent_map.keys())
    for p in parent_map.values():
        all_types.add(p)
        
    def get_descendants(t: str) -> Set[str]:
        desc = set()
        for child, parent in parent_map.items():
            if parent == t:
                desc.add(child)
                desc.update(get_descendants(child))
        return desc

    def get_prefix(name: str) -> str:
        match = re.match(r'^([a-zA-Z]+)', name)
        if match:
            return match.group(1).lower()
        return name.lower()

    refined = {}
    for obj, t in obj_types.items():
        obj_lower = obj.lower()
        t_lower = t.lower()
        
        subtypes = get_descendants(t_lower)
        subtypes.add(t_lower)
        
        # 1. Exact or substring match
        matched_type = None
        for sub in subtypes:
            if sub in obj_lower:
                if matched_type is None:
                    matched_type = sub
                else:
                    def is_subtype(t1, t2):
                        curr = t1
                        while curr in parent_map:
                            curr = parent_map[curr]
                            if curr == t2:
                                return True
                        return False
                    if is_subtype(sub, matched_type):
                        matched_type = sub
                        
        if matched_type:
            refined[obj] = matched_type
            continue

        # 2. Prefix match (e.g. t0 -> t -> truck)
        prefix = get_prefix(obj_lower)
        matched_type = None
        for sub in subtypes:
            if sub.startswith(prefix):
                if matched_type is None:
                    matched_type = sub
                else:
                    def is_subtype(t1, t2):
                        curr = t1
                        while curr in parent_map:
                            curr = parent_map[curr]
                            if curr == t2:
                                return True
                        return False
                    if is_subtype(sub, matched_type):
                        matched_type = sub
                        
        if matched_type:
            refined[obj] = matched_type
            continue

        # 3. Fallback matching globally in all types if prefix matches
        fallback_match = None
        for d_type in all_types:
            if d_type in obj_lower or d_type.startswith(prefix):
                if fallback_match is None:
                    fallback_match = d_type
                else:
                    def is_subtype(t1, t2):
                        curr = t1
                        while curr in parent_map:
                            curr = parent_map[curr]
                            if curr == t2:
                                return True
                        return False
                    if is_subtype(d_type, fallback_match):
                        fallback_match = d_type
        if fallback_match:
            refined[obj] = fallback_match
        else:
            refined[obj] = t
    return refined

def verify_plan_with_val(
    domain_name: str,
    example_id: str,
    initial_state_str: str,
    goal_state_str: str,
    plan: List[Any],
    debug: bool = False
) -> Dict[str, Any]:
    # Locate Validate binary
    bin_paths = [
        "VAL/build/macos64/Release/bin/Validate",
        "../VAL/build/macos64/Release/bin/Validate",
        "./VAL/build/macos64/Release/bin/Validate",
        "/Users/sarvesh/Desktop/IBM/IBM/VAL/build/macos64/Release/bin/Validate"
    ]
    validate_bin_path = None
    for p in bin_paths:
        if os.path.exists(p):
            validate_bin_path = p
            break
    if not validate_bin_path:
        validate_bin_path = "VAL/build/macos64/Release/bin/Validate"

    # 1. Parse states
    init_facts = []
    if initial_state_str.strip().startswith("[") and "predicate" in initial_state_str:
        try:
            init_facts_raw = json.loads(initial_state_str)
            init_facts = [(f["predicate"], f["args"]) for f in init_facts_raw]
        except Exception:
            init_facts = parse_facts_syntax(initial_state_str)
    else:
        init_facts = parse_facts_syntax(initial_state_str)

    goal_facts = []
    if goal_state_str.strip().startswith("[") and "predicate" in goal_state_str:
        try:
            goal_facts_raw = json.loads(goal_state_str)
            goal_facts = [(f["predicate"], f["args"]) for f in goal_facts_raw]
        except Exception:
            goal_facts = parse_facts_syntax(goal_state_str)
    else:
        goal_facts = parse_facts_syntax(goal_state_str)

    # 2. Parse actions
    parsed_actions = []
    if not isinstance(plan, list):
         if isinstance(plan, str):
             parsed_actions = parse_actions_syntax(plan)
    else:
         for action_item in plan:
             if isinstance(action_item, str):
                 parsed = parse_action_string(action_item)
                 if parsed:
                     parsed_actions.append(parsed)
             elif isinstance(action_item, dict):
                 name = action_item.get("name") or action_item.get("action")
                 args = action_item.get("args") or action_item.get("parameters")
                 if name:
                     parsed_actions.append((name, args or []))
             elif isinstance(action_item, (list, tuple)):
                 if len(action_item) >= 1:
                     parsed_actions.append((action_item[0], action_item[1] if len(action_item) > 1 else []))

    if debug:
        print(f"Parsed initial facts: {len(init_facts)}")
        print(f"Parsed goal facts: {len(goal_facts)}")
        print(f"Parsed actions: {len(parsed_actions)}")

    # 3. Load domain PDDL
    try:
        pddl_domain = get_pddl_domain(domain_name)
        pddl_domain = fix_pddl_domain_typing(pddl_domain)
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Could not load domain PDDL: {e}",
            "actions_applied": 0,
            "total_actions": len(parsed_actions)
        }

    # 4. Extract domain details
    pddl_domain_name = get_domain_name(pddl_domain)
    use_typing = ":typing" in pddl_domain.lower()
    
    pred_types = parse_predicates_signature(pddl_domain)
    parent_map = parse_types_hierarchy(pddl_domain)

    # Inject implicit facts for specific domains
    # 1. Swap and any domain with `not-eq` predicate:
    if "not-eq" in pred_types:
        all_objs = set()
        for pred, args in init_facts + goal_facts:
            for arg in args:
                all_objs.add(arg)
        has_not_eq = any(pred.lower() == "not-eq" for pred, _ in init_facts)
        if not has_not_eq:
            for obj1 in all_objs:
                for obj2 in all_objs:
                    if obj1 != obj2:
                        init_facts.append(("not-eq", [obj1, obj2]))

    # 2. Hanoi: populate `smaller` facts
    if domain_name.lower() == "hanoi":
        all_objs = set()
        for pred, args in init_facts + goal_facts:
            for arg in args:
                all_objs.add(arg)
        has_smaller = any(pred.lower() == "smaller" for pred, _ in init_facts)
        if not has_smaller:
            pegs = [o for o in all_objs if o.lower().startswith("peg")]
            disks = [o for o in all_objs if o.lower().startswith("d") and o[1:].isdigit()]
            for peg in pegs:
                for disk in disks:
                    init_facts.append(("smaller", [peg, disk]))
            for d1 in disks:
                for d2 in disks:
                    try:
                        i = int(d1[1:])
                        j = int(d2[1:])
                        if i > j:
                            init_facts.append(("smaller", [d1, d2]))
                    except ValueError:
                        pass
    
    # 5. Infer types and generate problem PDDL
    obj_types = infer_object_types(init_facts, goal_facts, pred_types, parent_map)
    obj_types = refine_types_by_name(obj_types, parent_map)
    pddl_problem = generate_problem_pddl(
        domain_name=pddl_domain_name,
        example_id=example_id,
        init_facts=init_facts,
        goal_facts=goal_facts,
        obj_types=obj_types,
        use_typing=use_typing
    )

    # 6. Format plan steps
    plan_lines = []
    for name, args in parsed_actions:
        name_clean = name.lower()
        args_clean = [a.lower() for a in args]
        if args_clean:
            plan_lines.append(f"({name_clean} {' '.join(args_clean)})")
        else:
            plan_lines.append(f"({name_clean})")
    plan_content = "\n".join(plan_lines)

    # 7. Write to temp files and run Validate
    with tempfile.TemporaryDirectory() as temp_dir:
        domain_file = os.path.join(temp_dir, "domain.pddl")
        problem_file = os.path.join(temp_dir, "problem.pddl")
        plan_file = os.path.join(temp_dir, "plan.plan")

        with open(domain_file, "w") as f:
            f.write(pddl_domain)
        with open(problem_file, "w") as f:
            f.write(pddl_problem)
        with open(plan_file, "w") as f:
            f.write(plan_content)

        if debug:
            print("--- PROBLEM ---")
            print(pddl_problem)

        cmd = [validate_bin_path, "-v", domain_file, problem_file, plan_file]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout = res.stdout
            stderr = res.stderr
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Execution of Validate failed: {e}",
                "actions_applied": 0,
                "total_actions": len(parsed_actions)
            }

        # 8. Parse VAL output
        if "Plan valid" in stdout:
            return {
                "valid": True,
                "reason": "Goal satisfied",
                "actions_applied": len(parsed_actions),
                "total_actions": len(parsed_actions)
            }
        
        if "Error in type-checking" in stdout or "Error in type-checking" in stderr:
            return {
                "valid": False,
                "reason": "Error: Error in type-checking!",
                "actions_applied": 0,
                "total_actions": len(parsed_actions)
            }

        if "unsatisfied precondition at time" in stdout or "unsatisfied precondition at time" in stderr:
            match = re.search(r"unsatisfied precondition at time (\d+)", stdout + stderr)
            step = 1
            if match:
                step = int(match.group(1))
            
            failed_action_match = re.search(r"Plan failed because of unsatisfied precondition in:\n([^\n]+)", stdout)
            failed_action_info = ""
            if failed_action_match:
                failed_action_info = failed_action_match.group(1).strip()
            
            reason = f"Action {step} failed: unsatisfied precondition"
            if failed_action_info:
                reason = f"Action {step} '{failed_action_info}' failed: unsatisfied precondition"
                
            return {
                "valid": False,
                "reason": reason,
                "actions_applied": max(0, step - 1),
                "total_actions": len(parsed_actions)
            }

        if "Goal not satisfied" in stdout:
            return {
                "valid": False,
                "reason": "Goal not satisfied",
                "actions_applied": len(parsed_actions),
                "total_actions": len(parsed_actions)
            }

        if "Plan failed to execute" in stdout:
            return {
                "valid": False,
                "reason": "Plan failed to execute",
                "actions_applied": 0,
                "total_actions": len(parsed_actions)
            }

        return {
            "valid": False,
            "reason": f"Validation failed. Output: {stdout.strip()[:100]}",
            "actions_applied": 0,
            "total_actions": len(parsed_actions)
        }

# ============================================================================
# MAIN EVALUATION SCRIPT
# ============================================================================

import argparse

def evaluate_baseline(custom_output_path: Optional[str] = None):
    input_path = custom_output_path or "baselines/zero_shot_results_action.json"
    output_path = custom_output_path or "baselines/zero_shot_results_action.json"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "r") as f:
        all_data = json.load(f)

    total_examples = 0
    total_correct = 0
    total_correct_with_plan = 0
    total_plan_verified = 0
    total_plan_possible = 0

    print(f"Starting evaluation of baseline results: {input_path} using VAL...")

    for domain, domain_data in all_data["domains"].items():
        print(f"\nProcessing baseline domain: {domain}")
        
        domain_data["total_examples"] = 0
        domain_data["correct_answer"] = 0
        domain_data["correct_with_plan"] = 0
        domain_data["plan_verified"] = 0
        domain_data["total_yes_examples"] = 0
        
        details = domain_data.get("details", [])
        if not details and "examples" in domain_data:
            details = domain_data["examples"]
            
        for ex in details:
            domain_data["total_examples"] += 1
            total_examples += 1
            
            gt_answer = str(ex.get("gt_answer") or ex.get("ground_truth_answer") or "no").lower()
            pred_answer = str(ex.get("pred_answer") or ex.get("final_answer") or "no").lower()
            
            is_correct = (gt_answer == pred_answer)
            if is_correct:
                domain_data["correct_answer"] += 1
                total_correct += 1
                
            is_gt_yes = (gt_answer == "yes")
            is_pred_yes = (pred_answer == "yes")
            
            plan_valid = False
            verification = {"valid": None, "reason": "N/A"}
            
            if is_pred_yes:
                plan_str = ex.get("generated_plan") or ex.get("plan")
                if plan_str and plan_str != "Not Applicable":
                    plan = [line.strip() for line in plan_str.split("\n") if line.strip()]
                    
                    init_state_raw = ex.get("generated_initial_state") or ex.get("predicted_initial_state") or ""
                    init_state = "\n".join([line for line in init_state_raw.split("\n") if not line.strip().startswith(";")])
                    
                    goal_state_raw = ex.get("generated_goal_state") or ex.get("predicted_goal_state") or ""
                    goal_state = "\n".join([line for line in goal_state_raw.split("\n") if not line.strip().startswith(";")])
                    
                    verify_res = verify_plan_with_val(
                        domain_name=domain,
                        example_id=ex.get("id") or ex.get("example_id") or f"{domain}_baseline",
                        initial_state_str=init_state,
                        goal_state_str=goal_state,
                        plan=plan
                    )
                    verification = verify_res
                    if verify_res["valid"]:
                        plan_valid = True
            
            if not is_pred_yes:
                ex["plan_valid"] = "N/A"
            else:
                ex["plan_valid"] = plan_valid
                
            ex["verification"] = verification
            
            if is_gt_yes:
                domain_data["total_yes_examples"] += 1
                total_plan_possible += 1
                if plan_valid:
                    domain_data["plan_verified"] += 1
                    total_plan_verified += 1
                    
            is_correct_with_plan = False
            if is_gt_yes:
                if is_pred_yes and plan_valid:
                    is_correct_with_plan = True
            else:
                if not is_pred_yes:
                    is_correct_with_plan = True
                    
            if is_correct_with_plan:
                domain_data["correct_with_plan"] += 1
                total_correct_with_plan += 1
                
            if not plan_valid:
                ex["plan_failure_reason"] = verification.get("reason", "Not verified")
            else:
                ex["plan_failure_reason"] = "Goal satisfied"
                
            if "executed_final_state" in ex:
                ex["executed_final_state"] = verification.get("executed_final_state") or "N/A"
                
        acc = domain_data["correct_answer"] / domain_data["total_examples"] if domain_data["total_examples"] else 0
        plan_acc = domain_data["plan_verified"] / domain_data["total_yes_examples"] if domain_data["total_yes_examples"] else 0
        
        domain_data["accuracy"] = acc
        domain_data["plan_accuracy"] = plan_acc
        
        print(f"  Examples: {domain_data['total_examples']}")
        print(f"  Accuracy: {acc:.2f}")
        print(f"  Plan Accuracy (on YES): {plan_acc:.2f}")

    overall_accuracy = total_correct / total_examples if total_examples else 0
    overall_accuracy_with_plan = total_correct_with_plan / total_examples if total_examples else 0
    overall_plan_accuracy = total_plan_verified / total_plan_possible if total_plan_possible else 0
    
    summary = {
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_with_plan": overall_accuracy_with_plan,
        "overall_plan_accuracy": overall_plan_accuracy,
        "total_examples": total_examples,
        "total_correct": total_correct,
        "total_correct_with_plan": total_correct_with_plan,
        "total_plan_verified": total_plan_verified,
        "total_plan_possible": total_plan_possible
    }
    
    all_data["summary"] = summary
    
    print("\nOverall Baseline Summary (with VAL):")
    print(json.dumps(summary, indent=2))
    
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)

def evaluate_search(custom_output_path: Optional[str] = None):
    output_path = custom_output_path or "evaluation/atlas_evaluation_results.json"
    src_dir = "src"
    pattern = os.path.join(src_dir, "*", "search_result_test_domain.json")
    files = glob.glob(pattern)
    
    if not files:
        print("No search_result_test_domain.json files found.")
        return

    all_results = {
        "summary": {},
        "domains": {}
    }

    # Global counters
    total_examples = 0
    total_correct = 0
    total_correct_with_plan = 0
    total_plan_verified = 0
    total_plan_possible = 0

    print(f"Found {len(files)} result files. Starting evaluation with VAL...")

    for file_path in sorted(files):
        domain = os.path.basename(os.path.dirname(file_path))
        print(f"\nProcessing domain: {domain}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
            
        domain_stats = {
            "total_examples": 0,
            "correct_answer": 0,
            "correct_with_plan": 0,
            "plan_verified": 0,
            "total_yes_examples": 0,
            "examples": []
        }
        
        for ex in data:
            domain_stats["total_examples"] += 1
            total_examples += 1
            
            # Extract fields
            gt_answer = str(ex.get("ground_truth_answer", "no")).lower()
            pred_answer = str(ex.get("final_answer", "no")).lower()
            plan = ex.get("plan", [])
            init_state = ex.get("predicted_initial_state", "")
            goal_state = ex.get("predicted_goal_state", "")
            
            is_correct = (gt_answer == pred_answer)
            if is_correct:
                domain_stats["correct_answer"] += 1
                total_correct += 1
                
            is_gt_yes = (gt_answer == "yes")
            is_pred_yes = (pred_answer == "yes")
            
            verification = {"valid": None, "reason": "N/A"}
            plan_valid = False
            
            if is_pred_yes:
                verify_res = verify_plan_with_val(
                    domain_name=domain,
                    example_id=ex.get("example_id", f"{domain}_test"),
                    initial_state_str=str(init_state),
                    goal_state_str=str(goal_state),
                    plan=plan
                )
                verification = verify_res
                if verify_res["valid"]:
                    plan_valid = True

            if is_gt_yes:
                domain_stats["total_yes_examples"] += 1
                total_plan_possible += 1
                if plan_valid:
                    domain_stats["plan_verified"] += 1
                    total_plan_verified += 1
            
            is_correct_with_plan = False
            if is_gt_yes:
                if is_pred_yes and plan_valid:
                    is_correct_with_plan = True
            else:
                 if not is_pred_yes:
                      is_correct_with_plan = True
                      
            if is_correct_with_plan:
                domain_stats["correct_with_plan"] += 1
                total_correct_with_plan += 1

            domain_stats["examples"].append({
                "id": ex.get("example_id"),
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
                "plan_valid": verification.get("valid"),
                "plan_reason": verification.get("reason"),
                "correct_with_plan": is_correct_with_plan
            })
            
        all_results["domains"][domain] = domain_stats
        
        # Calculate domain metrics
        acc = domain_stats["correct_answer"] / domain_stats["total_examples"] if domain_stats["total_examples"] else 0
        plan_acc = domain_stats["plan_verified"] / domain_stats["total_yes_examples"] if domain_stats["total_yes_examples"] else 0
        print(f"  Examples: {domain_stats['total_examples']}")
        print(f"  Accuracy: {acc:.2f}")
        print(f"  Plan Accuracy (on YES): {plan_acc:.2f}")

    # Aggregate summary
    overall_accuracy = total_correct / total_examples if total_examples else 0
    overall_accuracy_with_plan = total_correct_with_plan / total_examples if total_examples else 0
    overall_plan_accuracy = total_plan_verified / total_plan_possible if total_plan_possible else 0
    
    summary = {
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_with_plan": overall_accuracy_with_plan,
        "overall_plan_accuracy": overall_plan_accuracy,
        "total_examples": total_examples,
        "total_correct": total_correct,
        "total_correct_with_plan": total_correct_with_plan,
        "total_plan_verified": total_plan_verified,
        "total_plan_possible": total_plan_possible
    }
    
    all_results["summary"] = summary
    
    print("\nOverall Summary (with VAL):")
    print(json.dumps(summary, indent=2))
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Action Validator using KCL-Planning/VAL")
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline results instead of search results")
    parser.add_argument("--output", type=str, help="Custom output JSON path")
    args = parser.parse_args()

    if args.baseline:
        evaluate_baseline(args.output)
    else:
        evaluate_search(args.output)

if __name__ == "__main__":
    main()
