"""
Action Executor Module - PDDL-free action execution for all domains.

This module provides action execution logic without requiring PDDL files.
Preconditions and effects are manually implemented based on domain semantics.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

Fact = Tuple[str, Tuple[str, ...]]


# ============================================================================
# PARSING UTILITIES
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


# ============================================================================
# BASE EXECUTOR CLASS
# ============================================================================

class ActionExecutor:
    """Base class for domain-specific action executors."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.state: Set[Fact] = set()
        self.last_error: Optional[str] = None

    def set_state(self, facts: List[Tuple[str, List[str]]]) -> None:
        """Initialize executor state from parsed facts."""
        self.state = set()
        normalized = self.normalize_state(facts)
        for pred, args in normalized:
            self.state.add((pred, tuple(args)))

    def normalize_state(self, facts: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
        """Hook for domain-specific state normalization."""
        return facts

    def apply_action(self, action_name: str, args: List[str], facts: List[str] = None) -> bool:
        """Apply an action. Returns True if successful, False otherwise."""
        self.last_error = None
        # Support calling with just name/args if already parsed
        return self._execute_action(action_name.lower(), args, facts or [])

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        """Parse action string into name and args. Override in subclasses."""
        return self.parse_pddl_action(action_str)

    def parse_pddl_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        """Helper to parse action(args) or (action args)."""
        if not action_str: return None
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

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        """Override in subclasses to implement domain-specific actions."""
        self.last_error = f"Unknown action '{action_name}'"
        return False

    def check_goal(self, goal_facts: List[Tuple[str, List[str]]]) -> bool:
        """Check if goal is satisfied in current state."""
        normalized = self.normalize_state(goal_facts)
        for pred, args in normalized:
            if (pred, tuple(args)) not in self.state:
                return False
        return True

    def _has(self, pred: str, *args: str) -> bool:
        return (pred, tuple(args)) in self.state

    def _add(self, pred: str, *args: str) -> None:
        self.state.add((pred, tuple(args)))

    def _rm(self, pred: str, *args: str) -> None:
        self.state.discard((pred, tuple(args)))

    def _find_single(self, pred: str) -> Optional[str]:
        """Find a single-argument fact value."""
        for p, a in self.state:
            if p == pred and len(a) == 1:
                return a[0]
        return None


# ============================================================================
# BLOCKSWORLD EXECUTOR
# ============================================================================

class BlocksworldExecutor(ActionExecutor):
    """Blocksworld: pick-up, put-down, stack, unstack"""

    pddl_actions = {"pick-up", "put-down", "stack", "unstack"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "pick-up":
            return self._pickup(args, facts)
        elif action_name == "put-down":
            return self._putdown(args, facts)
        elif action_name == "stack":
            return self._stack(args, facts)
        elif action_name == "unstack":
            return self._unstack(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _pickup(self, args: List[str], facts: List[str]) -> bool:
        if not args: return False
        b = args[0]
        self._rm("clear", b)
        self._rm("ontable", b)
        self._rm("handempty")
        self._add("holding", b)
        return True

    def _putdown(self, args: List[str], facts: List[str]) -> bool:
        if not args: return False
        b = args[0]
        self._rm("holding", b)
        self._add("clear", b)
        self._add("ontable", b)
        self._add("handempty")
        return True

    def _stack(self, args: List[str], facts: List[str]) -> bool:
        # stack(?ob, ?underob)
        if len(args) < 2: return False
        b_top, b_under = args[0], args[1]
        self._rm("holding", b_top)
        self._rm("clear", b_under)
        self._add("handempty")
        self._add("clear", b_top)
        self._add("on", b_top, b_under)
        return True

    def _unstack(self, args: List[str], facts: List[str]) -> bool:
        # unstack(?ob, ?underob)
        if len(args) < 2: return False
        b_top, b_under = args[0], args[1]
        self._rm("handempty")
        self._rm("clear", b_top)
        self._rm("on", b_top, b_under)
        self._add("holding", b_top)
        self._add("clear", b_under)
        return True


# ============================================================================
# FERRY EXECUTOR
# ============================================================================

class FerryExecutor(ActionExecutor):
    """Ferry: board, debark, sail"""

    pddl_actions = {"board", "debark", "sail"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "board":
            return self._board(args, facts)
        elif action_name == "debark":
            return self._debark(args, facts)
        elif action_name == "sail":
            return self._sail(args, facts)
        return False
        
    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _board(self, args: List[str], facts: List[str]) -> bool:
        # board(?car, ?loc)
        if len(args) < 2: return False
        car, loc = args[0], args[1]
        self._rm("at", car, loc)
        self._rm("empty-ferry")
        self._add("on", car, "ferry")
        return True

    def _debark(self, args: List[str], facts: List[str]) -> bool:
        # debark(?car, ?loc)
        if len(args) < 2: return False
        car, loc = args[0], args[1]
        self._rm("on", car, "ferry")
        self._add("at", car, loc)
        self._add("empty-ferry")
        return True

    def _sail(self, args: List[str], facts: List[str]) -> bool:
        # sail(?from, ?to)
        if len(args) < 2: return False
        loc_from, loc_to = args[0], args[1]
        self._rm("at-ferry", loc_from)
        self._add("at-ferry", loc_to)
        return True


# ============================================================================
# LOGISTICS EXECUTOR
# ============================================================================

class LogisticsExecutor(ActionExecutor):
    """Logistics: drive-truck, fly-airplane, load-airplane, load-truck, unload-airplane, unload-truck"""

    pddl_actions = {
        "load-truck", "unload-truck", "drive-truck",
        "load-airplane", "unload-airplane", "fly-airplane"
    }

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "load-truck":
            return self._load_truck(args, facts)
        elif action_name == "unload-truck":
            return self._unload_truck(args, facts)
        elif action_name == "drive-truck":
            return self._drive_truck(args, facts)
        elif action_name == "load-airplane":
            return self._load_airplane(args, facts)
        elif action_name == "unload-airplane":
            return self._unload_airplane(args, facts)
        elif action_name == "fly-airplane":
            return self._fly_airplane(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _load_truck(self, args: List[str], facts: List[str]) -> bool:
        # load-truck(?obj, ?truck, ?loc)
        if len(args) < 3: return False
        obj, truck, loc = args[0], args[1], args[2]
        self._rm("at", obj, loc)
        self._add("in", obj, truck)
        return True

    def _unload_truck(self, args: List[str], facts: List[str]) -> bool:
        # unload-truck(?obj, ?truck, ?loc)
        if len(args) < 3: return False
        obj, truck, loc = args[0], args[1], args[2]
        self._rm("in", obj, truck)
        self._add("at", obj, loc)
        return True
        
    def _drive_truck(self, args: List[str], facts: List[str]) -> bool:
        # drive-truck(?truck, ?loc-from, ?loc-to, ?city)
        if len(args) < 3: return False
        truck, loc_from, loc_to = args[0], args[1], args[2]
        self._rm("at", truck, loc_from)
        self._add("at", truck, loc_to)
        return True

    def _load_airplane(self, args: List[str], facts: List[str]) -> bool:
        # load-airplane(?obj, ?airplane, ?loc)
        if len(args) < 3: return False
        obj, airplane, loc = args[0], args[1], args[2]
        self._rm("at", obj, loc)
        self._add("in", obj, airplane)
        return True

    def _unload_airplane(self, args: List[str], facts: List[str]) -> bool:
        # unload-airplane(?obj, ?airplane, ?loc)
        if len(args) < 3: return False
        obj, airplane, loc = args[0], args[1], args[2]
        self._rm("in", obj, airplane)
        self._add("at", obj, loc)
        return True
        
    def _fly_airplane(self, args: List[str], facts: List[str]) -> bool:
        # fly-airplane(?airplane, ?from, ?to)
        if len(args) < 3: return False
        plane, loc_from, loc_to = args[0], args[1], args[2]
        self._rm("at", plane, loc_from)
        self._add("at", plane, loc_to)
        return True


# ============================================================================
# GRIPPERS EXECUTOR
# ============================================================================

class GrippersExecutor(ActionExecutor):
    """Grippers: drop, move, pick"""

    pddl_actions = {"move", "pick", "drop"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "move":
            if len(args) == 3:
                return self._move(args, facts)
            # Standard "move" in PDDL is 3 args (robot, from, to)
            # If 2 args provided, it might be a simplification or parse issue
            # We strictly stick to PDDL structure generally
            return self._move(args, facts)
        elif action_name == "pick":
            return self._pick(args, facts)
        elif action_name == "drop":
            return self._drop(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _move(self, args: List[str], facts: List[str]) -> bool:
        # move(?robot, ?from, ?to) -- PDDL usually 3 args
        if len(args) < 3: return False
        robot, loc_from, loc_to = args[0], args[1], args[2]
        self._rm("at-robby", robot, loc_from)
        self._add("at-robby", robot, loc_to)
        return True
        
    def _pick(self, args: List[str], facts: List[str]) -> bool:
        # pick(?r, ?obj, ?room, ?g)
        if len(args) < 4: return False
        robot, ball, room, gripper = args[0], args[1], args[2], args[3]
        self._rm("at", ball, room)
        self._rm("free", robot, gripper)
        self._add("carry", robot, ball, gripper)
        return True

    def _drop(self, args: List[str], facts: List[str]) -> bool:
        # drop(?r, ?obj, ?room, ?g)
        if len(args) < 4: return False
        robot, ball, room, gripper = args[0], args[1], args[2], args[3]
        self._rm("carry", robot, ball, gripper)
        self._add("at", ball, room)
        self._add("free", robot, gripper)
        return True
        if not self._has("at-robby", robot, room):
            self.last_error = f"Robot {robot} not at {room}"
            return False
        self._rm("carry", robot, ball, gripper)
        self._add("at", ball, room)
        self._add("free", robot, gripper)
        return True


# ============================================================================
# VISITALL EXECUTOR
# ============================================================================

class VisitallExecutor(ActionExecutor):
    """Visitall: move"""

    pddl_actions = {"move"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "move":
            return self._move(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _move(self, args: List[str], facts: List[str]) -> bool:
        # move(?curpos, ?nextpos)
        if len(args) < 2: return False
        cur, next_pos = args[0], args[1]
        self._rm("at-robot", cur)
        self._add("at-robot", next_pos)
        self._add("visited", next_pos)
        return True


# ============================================================================
# GRID EXECUTOR
# ============================================================================

class GridExecutor(ActionExecutor):
    """Grid: move, pickup, pickup-and-loose, putdown, unlock"""

    pddl_actions = {"move", "pickup", "pickup-and-loose", "putdown", "unlock"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "move":
            return self._move(args, facts)
        elif action_name == "pickup":
            return self._pickup(args, facts)
        elif action_name == "pickup-and-loose":
            return self._pickup_and_loose(args, facts)
        elif action_name == "putdown":
            return self._putdown(args, facts)
        elif action_name == "unlock":
            return self._unlock(args, facts)
        return False
        
    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _move(self, args: List[str], facts: List[str]) -> bool:
        # move(?curpos, ?nextpos)
        if len(args) < 2: return False
        cur, next_pos = args[0], args[1]
        self._rm("at-robot", cur)
        self._add("at-robot", next_pos)
        return True

    def _pickup(self, args: List[str], facts: List[str]) -> bool:
        # pickup(?curpos, ?key)
        if len(args) < 2: return False
        cur, key = args[0], args[1]
        self._rm("at", key, cur)
        self._rm("arm-empty")
        self._add("holding", key)
        return True

    def _pickup_and_loose(self, args: List[str], facts: List[str]) -> bool:
        # pickup-and-loose(?curpos, ?newkey, ?oldkey)
        if len(args) < 3: return False
        cur, new_key, old_key = args[0], args[1], args[2]
        self._rm("at", new_key, cur)
        self._rm("holding", old_key)
        self._add("at", old_key, cur)
        self._add("holding", new_key)
        return True

    def _putdown(self, args: List[str], facts: List[str]) -> bool:
        # putdown(?curpos, ?key)
        if len(args) < 2: return False
        cur, key = args[0], args[1]
        self._rm("holding", key)
        self._add("at", key, cur)
        self._add("arm-empty")
        return True

    def _unlock(self, args: List[str], facts: List[str]) -> bool:
        # unlock(?curpos, ?lockpos, ?key, ?shape)
        if len(args) < 4: return False
        cur, lock_pos, key, shape = args[0], args[1], args[2], args[3]
        self._rm("locked", lock_pos)
        self._add("open", lock_pos)
        return True


# ============================================================================
# FLOORTILE EXECUTOR  
# ============================================================================

class FloortileExecutor(ActionExecutor):
    """Floortile: change-color, down, left, paint-down, paint-up, right, up"""

    pddl_actions = {"up", "down", "right", "left", "change-color", "paint-up", "paint-down"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "up":
            return self._move(args, facts, "up")
        elif action_name == "down":
            return self._move(args, facts, "down")
        elif action_name == "right":
            return self._move(args, facts, "right")
        elif action_name == "left":
            return self._move(args, facts, "left")
        elif action_name == "change-color":
            return self._change_color(args, facts)
        elif action_name == "paint-up":
            return self._paint(args, facts, "up")
        elif action_name == "paint-down":
            return self._paint(args, facts, "down")
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _move(self, args: List[str], facts: List[str], direction: str) -> bool:
        # up(?robot, ?from, ?to), same for down/left/right
        if len(args) < 3: return False
        robot, f, t = args[0], args[1], args[2]
        self._rm("at", robot, f)
        self._add("at", robot, t)
        return True

    def _change_color(self, args: List[str], facts: List[str]) -> bool:
        # change-color(?r, ?c, ?c2)
        if len(args) < 3: return False
        # Robot changes color
        return True

    def _paint(self, args: List[str], facts: List[str], direction: str) -> bool:
        # paint-up(?r, ?y, ?x, ?c) -> robot, pos-y, pos-x, color
        if len(args) < 4: return False
        robot, py, px, color = args[0], args[1], args[2], args[3]
        self._add("painted", py, color) # Simplified effect
        return True


# ============================================================================
# GOLDMINER EXECUTOR
# ============================================================================

class GoldminerExecutor(ActionExecutor):
    """Goldminer: detonate-bomb, fire-laser, move, pick-gold, pickup-bomb, pickup-laser, putdown-laser"""

    pddl_actions = {"detonate-bomb", "fire-laser", "move", "pick-gold", "pickup-bomb", "pickup-laser", "putdown-laser"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "move":
            return self._move(args, facts)
        elif action_name == "pickup-bomb":
            return self._pickup_bomb(args, facts)
        elif action_name == "pickup-laser":
            return self._pickup_laser(args, facts)
        elif action_name == "putdown-laser":
            return self._putdown_laser(args, facts)
        elif action_name == "fire-laser":
            return self._fire_laser(args, facts)
        elif action_name == "detonate-bomb":
            return self._detonate_bomb(args, facts)
        elif action_name == "pick-gold":
            return self._pick_gold(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _move(self, args: List[str], facts: List[str]) -> bool:
        # move(?from, ?to)
        if len(args) < 2: return False
        from_loc, to_loc = args[0], args[1]
        # Check robot is at from location
        if not self._has("robot-at", from_loc) and not self._has("at", from_loc):
            self.last_error = f"Robot not at {from_loc}"
            return False
        # Check connectivity or clear
        can_move = (self._has("clear", to_loc) or 
                    self._has("connected", from_loc, to_loc) or
                    not self._has_any_rock(to_loc))
        if not can_move:
            self.last_error = f"Location {to_loc} not clear"
            return False
        self._rm("robot-at", from_loc)
        self._rm("at", from_loc)
        self._add("robot-at", to_loc)
        return True

    def _has_any_rock(self, loc: str) -> bool:
        return (self._has("soft-rock-at", loc) or 
                self._has("hard-rock-at", loc) or
                self._has("soft-rock", loc) or
                self._has("hard-rock", loc))

    def _pickup_bomb(self, args: List[str], facts: List[str]) -> bool:
        # pickup-bomb(?loc)
        if not args: return False
        loc = args[0]
        if not self._has("robot-at", loc):
            self.last_error = f"Robot not at {loc}"
            return False
        if not self._has("bomb-at", loc):
            self.last_error = f"No bomb at {loc}"
            return False
        if not self._has("arm-empty"):
            self.last_error = "Arm not empty"
            return False
        self._rm("arm-empty")
        self._add("holds-bomb")
        return True

    def _pickup_laser(self, args: List[str], facts: List[str]) -> bool:
        # pickup-laser(?loc)
        if not args: return False
        loc = args[0]
        if not self._has("robot-at", loc):
            self.last_error = f"Robot not at {loc}"
            return False
        if not self._has("laser-at", loc):
            self.last_error = f"No laser at {loc}"
            return False
        if not self._has("arm-empty"):
            self.last_error = "Arm not empty"
            return False
        self._rm("arm-empty")
        self._rm("laser-at", loc)
        self._add("holds-laser")
        return True

    def _putdown_laser(self, args: List[str], facts: List[str]) -> bool:
        # putdown-laser(?loc)
        if not args: return False
        loc = args[0]
        if not self._has("holds-laser"):
            self.last_error = "Not holding laser"
            return False
        self._rm("holds-laser")
        self._add("laser-at", loc)
        self._add("arm-empty")
        return True

    def _fire_laser(self, args: List[str], facts: List[str]) -> bool:
        # fire-laser(?from, ?target)
        if len(args) < 2: return False
        from_loc, target = args[0], args[1]
        if not self._has("holds-laser"):
            self.last_error = "Not holding laser"
            return False
        # Destroy rocks
        if self._has("soft-rock-at", target):
            self._rm("soft-rock-at", target)
            self._add("clear", target)
        if self._has("hard-rock-at", target):
            self._rm("hard-rock-at", target)
            self._add("clear", target)
        return True

    def _detonate_bomb(self, args: List[str], facts: List[str]) -> bool:
        # detonate-bomb(?from, ?target)
        if len(args) < 2: return False
        from_loc, target = args[0], args[1]
        if not self._has("holds-bomb"):
            self.last_error = "Not holding bomb"
            return False
        if not self._has("soft-rock-at", target):
            self.last_error = f"No soft rock at {target}"
            return False
        self._rm("holds-bomb")
        self._rm("soft-rock-at", target)
        self._add("clear", target)
        self._add("arm-empty")
        return True

    def _pick_gold(self, args: List[str], facts: List[str]) -> bool:
        # pick-gold(?from, ?gold-loc)
        if len(args) < 2: return False
        from_loc, gold_loc = args[0], args[1]
        if not self._has("robot-at", from_loc):
            self.last_error = f"Robot not at {from_loc}"
            return False
        if from_loc != gold_loc:
             self.last_error = f"Robot location {from_loc} does not match gold location {gold_loc}"
             return False
        if not self._has("gold-at", gold_loc):
            self.last_error = f"No gold at {gold_loc}"
            return False
        if not self._has("arm-empty"):
            self.last_error = "Arm not empty"
            return False
        self._rm("gold-at", gold_loc)
        self._rm("arm-empty")
        self._add("holds-gold")
        return True


# ============================================================================
# DEPOT EXECUTOR
# ============================================================================

class DepotExecutor(ActionExecutor):
    """Depot: drive, drop, lift, load, unload"""

    pddl_actions = {"drive", "drop", "lift", "load", "unload"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "drive":
            return self._drive(args, facts)
        elif action_name == "lift":
            return self._lift(args, facts)
        elif action_name == "unload":
            return self._unload(args, facts)
        elif action_name == "drop":
            return self._drop(args, facts)
        elif action_name == "load":
            return self._load(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _drive(self, args: List[str], facts: List[str]) -> bool:
        # drive(?truck, ?from, ?to)
        if len(args) < 3: return False
        truck, from_loc, to_loc = args[0], args[1], args[2]
        self._rm("at", truck, from_loc)
        self._add("at", truck, to_loc)
        return True

    def _lift(self, args: List[str], facts: List[str]) -> bool:
        # lift(?hoist, ?crate, ?surface, ?loc)
        if len(args) < 4: return False
        hoist, crate, surface, loc = args[0], args[1], args[2], args[3]
        if not self._has("at", hoist, loc):
            self.last_error = f"Hoist {hoist} not at {loc}"
            return False
        if not self._has("available", hoist):
            self.last_error = f"Hoist {hoist} not available"
            return False
        if not self._has("at", crate, loc):
            self.last_error = f"Crate {crate} not at {loc}"
            return False
        if not self._has("on", crate, surface):
            self.last_error = f"Crate {crate} not on {surface}"
            return False
        self._rm("available", hoist)
        self._rm("on", crate, surface)
        self._rm("clear", crate)
        self._rm("at", crate, loc)
        self._add("lifting", hoist, crate)
        self._add("clear", surface)
        return True

    def _unload(self, args: List[str], facts: List[str]) -> bool:
        # unload(?hoist, ?crate, ?truck, ?loc)
        if len(args) < 4: return False
        hoist, crate, truck, loc = args[0], args[1], args[2], args[3]
        if not self._has("at", hoist, loc):
            self.last_error = f"Hoist {hoist} not at {loc}"
            return False
        if not self._has("available", hoist):
            self.last_error = f"Hoist {hoist} not available"
            return False
        if not self._has("in", crate, truck):
            self.last_error = f"Crate {crate} not in {truck}"
            return False
        self._rm("available", hoist)
        self._rm("in", crate, truck)
        self._add("lifting", hoist, crate)
        return True

    def _drop(self, args: List[str], facts: List[str]) -> bool:
        # drop(?hoist, ?crate, ?surface, ?loc)
        if len(args) < 4: return False
        hoist, crate, surface, loc = args[0], args[1], args[2], args[3]
        if not self._has("lifting", hoist, crate):
            self.last_error = f"Hoist {hoist} not lifting {crate}"
            return False
        if not self._has("clear", surface):
            self.last_error = f"Surface {surface} not clear"
            return False
        self._rm("lifting", hoist, crate)
        self._rm("clear", surface)
        self._add("on", crate, surface)
        self._add("clear", crate)
        self._add("available", hoist)
        self._add("at", crate, loc)
        return True

    def _load(self, args: List[str], facts: List[str]) -> bool:
        # load(?hoist, ?crate, ?truck, ?loc)
        if len(args) < 4: return False
        hoist, crate, truck, loc = args[0], args[1], args[2], args[3]
        if not self._has("lifting", hoist, crate):
            self.last_error = f"Hoist {hoist} not lifting {crate}"
            return False
        if not self._has("at", truck, loc):
            self.last_error = f"Truck {truck} not at {loc}"
            return False
        self._rm("lifting", hoist, crate)
        self._add("in", crate, truck)
        self._add("available", hoist)
        return True


# ============================================================================
# ROVERS EXECUTOR
# ============================================================================

class RoversExecutor(ActionExecutor):
    """Rovers: calibrate, communicate_image_data, communicate_rock_data, communicate_soil_data, drop, navigate, sample_rock, sample_soil, take_image"""

    pddl_actions = {
        "navigate", "sample_soil", "sample_rock", "drop", "calibrate", 
        "take_image", "communicate_soil_data", "communicate_rock_data", "communicate_image_data"
    }

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "navigate":
            return self._navigate(args, facts)
        elif action_name == "sample_soil":
            return self._sample_soil(args, facts)
        elif action_name == "sample_rock":
            return self._sample_rock(args, facts)
        elif action_name == "drop":
            return self._drop(args, facts)
        elif action_name == "calibrate":
            return self._calibrate(args, facts)
        elif action_name == "take_image":
            return self._take_image(args, facts)
        elif action_name.startswith("communicate_"):
            if action_name == "communicate_soil_data":
                return self._communicate_soil(args, facts)
            elif action_name == "communicate_rock_data":
                return self._communicate_rock(args, facts)
            elif action_name == "communicate_image_data":
                return self._communicate_image(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _navigate(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 3: return False
        rover, from_wp, to_wp = args[0], args[1], args[2]
        if not self._has("at", rover, from_wp):
            self.last_error = f"Rover {rover} not at {from_wp}"
            return False
        if not self._has("can_traverse", rover, from_wp, to_wp):
            self.last_error = f"Cannot traverse from {from_wp} to {to_wp}"
            return False
        self._rm("at", rover, from_wp)
        self._add("at", rover, to_wp)
        return True

    def _sample_soil(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 3: return False
        rover, store, wp = args[0], args[1], args[2]
        if not self._has("at", rover, wp):
            self.last_error = f"Rover {rover} not at {wp}"
            return False
        if not self._has("at_soil_sample", wp):
            self.last_error = f"No soil sample at {wp}"
            return False
        if not self._has("equipped_for_soil_analysis", rover):
            self.last_error = f"Rover {rover} not equipped for soil"
            return False
        if not self._has("store_of", store, rover):
            self.last_error = f"Store {store} not of rover {rover}"
            return False
        if not self._has("empty", store):
            self.last_error = f"Store {store} not empty"
            return False
        self._rm("empty", store)
        self._rm("at_soil_sample", wp)
        self._add("full", store)
        self._add("have_soil_analysis", rover, wp)
        return True

    def _sample_rock(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 3: return False
        rover, store, wp = args[0], args[1], args[2]
        if not self._has("at", rover, wp):
            self.last_error = f"Rover {rover} not at {wp}"
            return False
        if not self._has("at_rock_sample", wp):
            self.last_error = f"No rock sample at {wp}"
            return False
        if not self._has("equipped_for_rock_analysis", rover):
            self.last_error = f"Rover {rover} not equipped for rock"
            return False
        if not self._has("store_of", store, rover):
            self.last_error = f"Store {store} not of rover {rover}"
            return False
        if not self._has("empty", store):
            self.last_error = f"Store {store} not empty"
            return False
        self._rm("empty", store)
        self._rm("at_rock_sample", wp)
        self._add("full", store)
        self._add("have_rock_analysis", rover, wp)
        return True

    def _drop(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 2: return False
        rover, store = args[0], args[1]
        if not self._has("store_of", store, rover):
            self.last_error = f"Store {store} not of rover {rover}"
            return False
        if not self._has("full", store):
            self.last_error = f"Store {store} not full"
            return False
        self._rm("full", store)
        self._add("empty", store)
        return True

    def _calibrate(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 4: return False
        rover, camera, objective, wp = args[0], args[1], args[2], args[3]
        if not self._has("equipped_for_imaging", rover):
            self.last_error = f"Rover {rover} not equipped for imaging"
            return False
        if not self._has("on_board", camera, rover):
            self.last_error = f"Camera {camera} not on {rover}"
            return False
        if not self._has("at", rover, wp):
            self.last_error = f"Rover {rover} not at {wp}"
            return False
        self._add("calibrated", camera, rover)
        return True

    def _take_image(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 4: return False
        rover, wp, objective, camera = args[0], args[1], args[2], args[3]
        if not self._has("calibrated", camera, rover):
            self.last_error = f"Camera {camera} not calibrated"
            return False
        if not self._has("on_board", camera, rover):
            self.last_error = f"Camera {camera} not on {rover}"
            return False
        if not self._has("at", rover, wp):
            self.last_error = f"Rover {rover} not at {wp}"
            return False
        self._add("have_image", rover, objective)
        self._rm("calibrated", camera, rover)
        return True

    def _communicate_soil(self, args: List[str], facts: List[str]) -> bool:
        # communicate_soil_data(?r, ?l, ?p, ?x, ?y)
        if len(args) < 5: return False
        # ... implementation simplified ...
        self._add("communicated_soil_data", args[2])
        return True

    def _communicate_rock(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 5: return False
        self._add("communicated_rock_data", args[2])
        return True

    def _communicate_image(self, args: List[str], facts: List[str]) -> bool:
        if len(args) < 6: return False
        self._add("communicated_image_data", args[2], args[3])
        return True


# ============================================================================
# SATELLITE EXECUTOR
# ============================================================================

class SatelliteExecutor(ActionExecutor):
    """Satellite: calibrate, switch_off, switch_on, take_image, turn_to"""

    pddl_actions = {"turn_to", "switch_on", "switch_off", "calibrate", "take_image"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "turn_to":
            return self._turn_to(args, facts)
        elif action_name == "switch_on":
            return self._switch_on(args, facts)
        elif action_name == "switch_off":
            return self._switch_off(args, facts)
        elif action_name == "calibrate":
            return self._calibrate(args, facts)
        elif action_name == "take_image":
            return self._take_image(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _switch_on(self, args: List[str], facts: List[str]) -> bool:
        # switch_on(?i, ?s) -> instr, sat
        if len(args) < 2: return False
        instrument, satellite = args[0], args[1]
        if not self._has("on_board", instrument, satellite):
            self.last_error = f"Instrument {instrument} not on {satellite}"
            return False
        if not self._has("power_avail", satellite):
            self.last_error = f"No power on {satellite}"
            return False
        self._rm("power_avail", satellite)
        self._add("power_on", instrument)
        return True

    def _switch_off(self, args: List[str], facts: List[str]) -> bool:
        # switch_off(?i, ?s) -> instr, sat
        if len(args) < 2: return False
        instrument, satellite = args[0], args[1]
        if not self._has("on_board", instrument, satellite):
            self.last_error = f"Instrument {instrument} not on {satellite}"
            return False
        if not self._has("power_on", instrument):
            self.last_error = f"Instrument {instrument} not powered on"
            return False
        self._rm("power_on", instrument)
        self._add("power_avail", satellite)
        return True

    def _calibrate(self, args: List[str], facts: List[str]) -> bool:
        # calibrate(?s, ?i, ?d) -> sat, instr, dir
        if len(args) < 3: return False
        satellite, instrument, direction = args[0], args[1], args[2]
        if not self._has("on_board", instrument, satellite):
            self.last_error = f"Instrument {instrument} not on {satellite}"
            return False
        if not self._has("power_on", instrument):
            self.last_error = f"Instrument {instrument} not powered on"
            return False
        if not self._has("pointing", satellite, direction):
            self.last_error = f"Satellite {satellite} not pointing at {direction}"
            return False
        self._add("calibrated", instrument)
        return True

    def _take_image(self, args: List[str], facts: List[str]) -> bool:
        # take_image(?s, ?d, ?i, ?m) -> sat, dir, instr, mode
        if len(args) < 4: return False
        satellite, direction, instrument, mode = args[0], args[1], args[2], args[3]
        if not self._has("calibrated", instrument):
            self.last_error = f"Instrument {instrument} not calibrated"
            return False
        if not self._has("power_on", instrument):
            self.last_error = f"Instrument {instrument} not powered on"
            return False
        if not self._has("pointing", satellite, direction):
            self.last_error = f"Satellite {satellite} not pointing at {direction}"
            return False
        if not self._has("on_board", instrument, satellite):
            self.last_error = f"Instrument {instrument} not on {satellite}"
            return False
        self._add("have_image", direction, mode)
        return True

    def _turn_to(self, args: List[str], facts: List[str]) -> bool:
        # turn_to(?s, ?d_new, ?d_prev) -> sat, new, old
        if len(args) < 3: return False
        satellite, new_dir, old_dir = args[0], args[1], args[2]
        if not self._has("pointing", satellite, old_dir):
            self.last_error = f"Satellite {satellite} not pointing at {old_dir}"
            return False
        self._rm("pointing", satellite, old_dir)
        self._add("pointing", satellite, new_dir)
        return True


# ============================================================================
# SWAP EXECUTOR
# ============================================================================

class SwapExecutor(ActionExecutor):
    """Swap: swap"""

    pddl_actions = {"swap"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "swap":
            return self._swap(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _swap(self, args: List[str], facts: List[str]) -> bool:
        # swap(?x, ?y)
        if len(args) < 2: return False
        x, y = args[0], args[1]
        if not self._has("adj", x, y):
            self.last_error = f"Positions {x} and {y} not adjacent"
            return False
        # Find what's at each position
        x_val = None
        y_val = None
        for p, a in list(self.state):
            if p == "at" and len(a) == 2:
                if a[1] == x:
                    x_val = a[0]
                elif a[1] == y:
                    y_val = a[0]
        if x_val and y_val:
            self._rm("at", x_val, x)
            self._rm("at", y_val, y)
            self._add("at", x_val, y)
            self._add("at", y_val, x)
        return True


# ============================================================================
# ALFWORLD EXECUTOR (Simplified)
# ============================================================================

class AlfworldExecutor(ActionExecutor):
    """Alfworld: clean_object, close_receptacle, cool_object, go_to_location, heat_object, open_receptacle, pickup_object_from_not_openable_receptacle, pickup_object_from_openable_receptacle, put_object_in_openable_receptacle, put_object_on_not_openable_receptacle, slice_object, toggle_object_off, toggle_object_on, validate_*"""

    def try_parse_action(self, action_name: str, args: List[str]) -> Tuple[str, List[str]]:
        # Normalize action names - replace underscores with hyphens
        name = action_name.lower().replace("_", "-")
        return name, args

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        """
        Execute an Alfworld action with strict PDDL name matching.
        """
        # Dispatch table for strict PDDL action names
        dispatch = {
            "go_to_location": self._go_to,
            "open_receptacle": self._open,
            "close_receptacle": self._close,
            "pickup_object_from_not_openable_receptacle": self._pickup,
            "pickup_object_from_openable_receptacle": self._pickup,
            "put_object_in_openable_receptacle": self._put,
            "put_object_on_not_openable_receptacle": self._put,
            "clean_object": self._clean,
            "cool_object": self._cool,
            "heat_object": self._heat,
            "slice_object": self._slice,
            "toggle_object_on": self._toggle_on,
            "toggle_object_off": self._toggle_off,
            "validate_clean_and_place_in_receptacle": lambda a, f: self._validate(a, "clean"),
            "validate_cool_and_place_in_receptacle": lambda a, f: self._validate(a, "cool"),
            "validate_heat_and_place_in_receptacle": lambda a, f: self._validate(a, "heat"),
            "validate_pick_and_place_in_receptacle": lambda a, f: self._validate(a, "place"),
            "validate_pick_two_and_place_in_receptacle": lambda a, f: self._validate(a, "two_place"),
            "validate_examine_in_light": lambda a, f: self._validate(a, "examine"),
        }

        handler = dispatch.get(action_name)
        if handler:
            return handler(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str:
            return None
            
        # Standard PDDL parsing: action(arg1, arg2, ...) or (action arg1 arg2 ...)
        # Handling action(params) format
        match = re.match(r"(\w+)\((.*)\)", action_str.strip())
        if match:
            name = match.group(1)
            args_str = match.group(2)
            args = [a.strip() for a in args_str.split(",") if a.strip()]
            return name, args
            
        # Handling (action arg1 arg2) format
        if action_str.startswith("(") and action_str.endswith(")"):
            content = action_str[1:-1].strip()
            parts = content.split()
            if parts:
                return parts[0], parts[1:]
                
        return None

    def _go_to(self, args: List[str], facts: List[str]) -> bool:
        """Move agent to location. Args can be [agent, from, to, recep] or [loc]."""
        if not args:
            self.last_error = "No location specified"
            return False
        # Get destination - could be last arg or only arg
        loc = args[-1] if len(args) >= 2 else args[0]
        # Remove old location
        for p, a in list(self.state):
            if p in ("agent-at", "atlocation", "at-location"):
                self.state.remove((p, a))
        self._add("atlocation", args[0] if len(args) >= 4 else "agent1", loc)
        return True

    def _open(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        recep = args[-1] if len(args) > 1 else args[0]
        self._rm("closed", recep)
        self._add("is-open", recep)
        self._add("opened", recep)
        return True

    def _close(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        recep = args[-1] if len(args) > 1 else args[0]
        self._rm("is-open", recep)
        self._rm("opened", recep)
        self._add("closed", recep)
        return True

    def _pickup(self, args: List[str], facts: List[str]) -> bool:
        """Pick up object. Args vary by action variant."""
        if not args:
            return False
        # Object is typically first or second arg
        obj = args[1] if len(args) >= 2 else args[0]
        # Remove object from any location/receptacle
        for p, a in list(self.state):
            if p in ("obj-at", "in-receptacle", "inreceptacle", "objectatlocation") and len(a) >= 1 and a[0] == obj:
                self.state.remove((p, a))
        self._add("holding", obj)
        return True

    def _put(self, args: List[str], facts: List[str]) -> bool:
        """Put object in receptacle. Args vary by action variant."""
        if len(args) < 2:
            self.last_error = "Need object and receptacle"
            return False
        obj = args[1] if len(args) >= 3 else args[0]
        recep = args[-1]
        self._rm("holding", obj)
        self._add("inreceptacle", obj, recep)
        return True

    def _clean(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        obj = args[0]
        self._add("is-clean", obj)
        self._add("cleaned", obj)
        return True

    def _cool(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        obj = args[0]
        self._add("is-cool", obj)
        return True

    def _heat(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        obj = args[0]
        self._add("is-hot", obj)
        return True

    def _slice(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        obj = args[0]
        self._add("is-sliced", obj)
        return True

    def _toggle_on(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        obj = args[0]
        self._rm("turned-off", obj)
        self._add("turned-on", obj)
        return True

    def _toggle_off(self, args: List[str], facts: List[str]) -> bool:
        if not args:
            return False
        obj = args[0]
        self._rm("turned-on", obj)
        self._add("turned-off", obj)
        return True

    def _validate(self, args: List[str], type_str: str) -> bool:
        """Handle validation actions."""
        # Validation just marks as validated
        self._add("validated")
        return True



# ============================================================================
# FROGS JUMPING EXECUTOR
# ============================================================================

class FrogsJumpingExecutor(ActionExecutor):
    """FrogsJumping: jump-left, jump-right, slide-left, slide-right"""

    pddl_actions = {"jump-left", "jump-right", "slide-left", "slide-right"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "jump-left":
            return self._jump_left(args, facts)
        elif action_name == "jump-right":
            return self._jump_right(args, facts)
        elif action_name == "slide-left":
            return self._slide_left(args, facts)
        elif action_name == "slide-right":
            return self._slide_right(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _jump_left(self, args: List[str], facts: List[str]) -> bool:
        # jump-left(?rf, ?from, ?mid, ?to, ?lf)
        if len(args) < 5: return False
        rf, loc_from, loc_mid, loc_to, lf = args[0], args[1], args[2], args[3], args[4]
        
        # Preconditions
        if not self._has("at", rf, loc_from):
            self.last_error = f"Frog {rf} not at {loc_from}"
            return False
        if not self._has("at", lf, loc_mid):
            self.last_error = f"Frog {lf} not at {loc_mid}"
            return False
        if not self._has("empty", loc_to):
            self.last_error = f"Location {loc_to} not empty"
            return False
        if not self._has("next", loc_to, loc_mid):
            self.last_error = f"{loc_to} is not next to {loc_mid}"
            return False
        if not self._has("next", loc_mid, loc_from):
            self.last_error = f"{loc_mid} is not next to {loc_from}"
            return False
            
        # Effects
        self._rm("at", rf, loc_from)
        self._add("at", rf, loc_to)
        self._add("empty", loc_from)
        self._rm("empty", loc_to)
        return True

    def _jump_right(self, args: List[str], facts: List[str]) -> bool:
        # jump-right(?lf, ?from, ?mid, ?to, ?rf)
        if len(args) < 5: return False
        lf, loc_from, loc_mid, loc_to, rf = args[0], args[1], args[2], args[3], args[4]
        
        # Preconditions
        if not self._has("at", lf, loc_from):
            self.last_error = f"Frog {lf} not at {loc_from}"
            return False
        if not self._has("at", rf, loc_mid):
            self.last_error = f"Frog {rf} not at {loc_mid}"
            return False
        if not self._has("empty", loc_to):
            self.last_error = f"Location {loc_to} not empty"
            return False
        if not self._has("next", loc_from, loc_mid):
            self.last_error = f"{loc_from} is not next to {loc_mid}"
            return False
        if not self._has("next", loc_mid, loc_to):
            self.last_error = f"{loc_mid} is not next to {loc_to}"
            return False
            
        # Effects
        self._rm("at", lf, loc_from)
        self._add("at", lf, loc_to)
        self._add("empty", loc_from)
        self._rm("empty", loc_to)
        return True

    def _slide_left(self, args: List[str], facts: List[str]) -> bool:
        # slide-left(?rf, ?from, ?to)
        if len(args) < 3: return False
        rf, loc_from, loc_to = args[0], args[1], args[2]
        
        # Preconditions
        if not self._has("at", rf, loc_from):
            self.last_error = f"Frog {rf} not at {loc_from}"
            return False
        if not self._has("empty", loc_to):
            self.last_error = f"Location {loc_to} not empty"
            return False
        if not self._has("next", loc_to, loc_from):
            self.last_error = f"{loc_to} is not next to {loc_from}"
            return False
            
        # Effects
        self._rm("at", rf, loc_from)
        self._add("at", rf, loc_to)
        self._add("empty", loc_from)
        self._rm("empty", loc_to)
        return True

    def _slide_right(self, args: List[str], facts: List[str]) -> bool:
        # slide-right(?lf, ?from, ?to)
        if len(args) < 3: return False
        lf, loc_from, loc_to = args[0], args[1], args[2]
        
        # Preconditions
        if not self._has("at", lf, loc_from):
            self.last_error = f"Frog {lf} not at {loc_from}"
            return False
        if not self._has("empty", loc_to):
            self.last_error = f"Location {loc_to} not empty"
            return False
        if not self._has("next", loc_from, loc_to):
            self.last_error = f"{loc_from} is not next to {loc_to}"
            return False
            
        # Effects
        self._rm("at", lf, loc_from)
        self._add("at", lf, loc_to)
        self._add("empty", loc_from)
        self._rm("empty", loc_to)
        return True


# ============================================================================
# HANOI EXECUTOR
# ============================================================================

class HanoiExecutor(ActionExecutor):
    """Hanoi: move"""

    pddl_actions = {"move"}

    def _execute_action(self, action_name: str, args: List[str], facts: List[str]) -> bool:
        if action_name == "move":
            return self._move(args, facts)
        return False

    def try_parse_action(self, action_str: str) -> Optional[Tuple[str, List[str]]]:
        if not action_str: return None
        parsed = self.parse_pddl_action(action_str)
        if parsed and parsed[0] in self.pddl_actions:
            return parsed
        return None

    def _move(self, args: List[str], facts: List[str]) -> bool:
        # move(?disk, ?from, ?to)
        if len(args) < 3: return False
        disk, loc_from, loc_to = args[0], args[1], args[2]
        
        # Preconditions
        # (smaller ?to ?disk)
        if not self._has("smaller", loc_to, disk):
            self.last_error = f"{loc_to} is not acceptable for {disk} (smaller constraint)"
            return False
        # (on ?disk ?from)
        if not self._has("on", disk, loc_from):
            self.last_error = f"{disk} is not on {loc_from}"
            return False
        # (clear ?disk)
        if not self._has("clear", disk):
            self.last_error = f"{disk} is not clear"
            return False
        # (clear ?to)
        if not self._has("clear", loc_to):
            self.last_error = f"{loc_to} is not clear"
            return False
            
        # Effects
        self._add("clear", loc_from)
        self._add("on", disk, loc_to)
        self._rm("on", disk, loc_from)
        self._rm("clear", loc_to)
        return True


# ============================================================================
# EXECUTOR FACTORY
# ============================================================================

EXECUTOR_MAP: Dict[str, type] = {
    "blocksworld": BlocksworldExecutor,
    "blocksworld-4ops": BlocksworldExecutor,
    "ferry": FerryExecutor,
    "logistics": LogisticsExecutor,
    "grippers": GrippersExecutor,
    "rovers": RoversExecutor,
    "visitall": VisitallExecutor,
    "grid": GridExecutor,
    "floortile": FloortileExecutor,
    "alfworld": AlfworldExecutor,
    "depot": DepotExecutor,
    "goldminer": GoldminerExecutor,
    "satellite": SatelliteExecutor,
    "swap": SwapExecutor,
    "frogs-jumping": FrogsJumpingExecutor,
    "frogs_jumping": FrogsJumpingExecutor,
    "hanoi": HanoiExecutor,
}


def get_executor(domain_name: str) -> ActionExecutor:
    """Get the appropriate executor for the given domain."""
    domain_lower = domain_name.lower().replace("_", "-")
    
    # Check for exact match
    if domain_lower in EXECUTOR_MAP:
        return EXECUTOR_MAP[domain_lower](domain_lower)
    
    # Check for prefix match
    for key, executor_class in EXECUTOR_MAP.items():
        if domain_lower.startswith(key) or key.startswith(domain_lower):
            return executor_class(domain_lower)
    
    raise ValueError(f"Unknown domain: {domain_name}")
