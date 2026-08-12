"""
Unified PDDL Parser for all test case generators.

This module provides a robust, consistent PDDL parsing solution that can be
used across all three test case generators (nl2state, succ, goal) with proper
fallback mechanisms and error handling.
"""
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict

# Type alias for S-expression parsing
Token = Union[str, List['Token']]


class UnifiedPDDLParser:
    """
    A unified PDDL parser that provides consistent parsing across all generators.
    
    Uses tarski when available, with reliable fallback to S-expression parsing.
    Handles domain name extraction, state parsing, and object extraction uniformly.
    """
    
    def __init__(self):
        """Initialize the parser with tarski availability check."""
        self.tarski_available = False
        self.PDDLReader = None
        
        try:
            from tarski.io import PDDLReader
            self.PDDLReader = PDDLReader
            self.tarski_available = True
        except ImportError:
            self.tarski_available = False
    
    # ---------------------------
    # S-Expression Parser Methods
    # ---------------------------
    
    @staticmethod
    def tokenize(pddl: str) -> List[str]:
        """Tokenize PDDL string into list of tokens."""
        # Remove comments
        pddl = re.sub(r";;?.*$", "", pddl, flags=re.MULTILINE)
        pddl = pddl.replace("\t", " ")
        # Find parentheses and non-whitespace sequences
        return re.findall(r"\(|\)|[^\s()]+", pddl)
    
    @classmethod
    def parse_tokens(cls, tokens: List[str], idx: int = 0) -> Tuple[Token, int]:
        """Parse tokenized PDDL into S-expression tree."""
        lst: List[Token] = []
        if idx >= len(tokens) or tokens[idx] != "(":
            raise ValueError(f"Expected '(' at position {idx}")
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
        """Parse PDDL string into S-expression."""
        try:
            tokens = cls.tokenize(pddl)
            if not tokens:
                raise ValueError("Empty PDDL content")
            node, _ = cls.parse_tokens(tokens, 0)
            return node
        except Exception as e:
            raise ValueError(f"Failed to parse PDDL S-expression: {e}")
    
    # ---------------------------
    # Domain Parsing Methods
    # ---------------------------
    
    @staticmethod
    def find_domain_name(domain_root: Token) -> str:
        """Extract domain name from parsed PDDL domain."""
        if not isinstance(domain_root, list) or len(domain_root) < 2:
            return "unknown-domain"
        
        # Look for (define (domain NAME) ...) pattern
        def search_domain_name(node: Token) -> Optional[str]:
            if isinstance(node, list):
                if len(node) >= 3 and node[0] == "define" and isinstance(node[1], list):
                    define_block = node[1]
                    if len(define_block) >= 2 and define_block[0] == "domain":
                        return str(define_block[1])
                
                # Recursively search in sublists
                for item in node:
                    result = search_domain_name(item)
                    if result:
                        return result
            return None
        
        domain_name = search_domain_name(domain_root)
        return domain_name if domain_name else "unknown-domain"
    
    @staticmethod
    def extract_objects(problem_root: Token) -> Dict[str, List[str]]:
        """Extract objects from parsed PDDL problem, grouped by type."""
        type_to_objs: Dict[str, List[str]] = defaultdict(list)
        
        def walk(root: Token):
            if isinstance(root, list) and len(root) > 0 and root[0] == ":objects":
                items = root[1:]
                tmp: List[str] = []
                i = 0
                
                while i < len(items):
                    it = items[i]
                    if it == "-":
                        # Type specification
                        if i + 1 < len(items):
                            type_name = str(items[i + 1])
                            for obj in tmp:
                                type_to_objs[type_name].append(obj)
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
                
                # Add remaining objects as untyped
                for obj in tmp:
                    type_to_objs["object"].append(obj)
            
            elif isinstance(root, list):
                for item in root:
                    walk(item)
        
        walk(problem_root)
        return dict(type_to_objs)
    
    # ---------------------------
    # State Parsing Methods
    # ---------------------------
    
    def extract_state_predicates(self, problem_root: Token) -> List[Tuple[str, List[str]]]:
        """Extract state predicates from PDDL problem initial state."""
        predicates: List[Tuple[str, List[str]]] = []
        
        def walk(root: Token):
            if isinstance(root, list) and len(root) > 0 and root[0] == ":init":
                for item in root[1:]:
                    if isinstance(item, list) and item:
                        pred_name = str(item[0])
                        args = [str(arg) for arg in item[1:]]
                        predicates.append((pred_name, args))
                    elif isinstance(item, str) and item.strip():
                        # Handle zero-arity predicates
                        predicates.append((item, []))
            elif isinstance(root, list):
                for item in root:
                    walk(item)
        
        walk(problem_root)
        return predicates
    
    # ---------------------------
    # High-level Parsing Methods
    # ---------------------------
    
    def parse_pddl_domain(self, domain_pddl: str) -> Tuple[str, Token]:
        """Parse PDDL domain and return domain name and parsed tree."""
        try:
            domain_root = self.sexpr(domain_pddl)
            domain_name = self.find_domain_name(domain_root)
            return domain_name, domain_root
        except Exception as e:
            raise ValueError(f"Failed to parse PDDL domain: {e}")
    
    def parse_pddl_problem(self, problem_pddl: str) -> Tuple[Dict[str, List[str]], List[Tuple[str, List[str]]], Token]:
        """Parse PDDL problem and return objects, initial state, and parsed tree."""
        try:
            problem_root = self.sexpr(problem_pddl)
            objects = self.extract_objects(problem_root)
            init_state = self.extract_state_predicates(problem_root)
            return objects, init_state, problem_root
        except Exception as e:
            raise ValueError(f"Failed to parse PDDL problem: {e}")
    
    def validate_with_tarski(self, domain_pddl: str, problem_pddl: str) -> bool:
        """Validate PDDL files using tarski if available."""
        if not self.tarski_available or not self.PDDLReader:
            return True  # Skip validation if tarski unavailable
        
        try:
            with tempfile.TemporaryDirectory() as td:
                dom_path = Path(td) / "domain.pddl"
                prob_path = Path(td) / "problem.pddl"
                dom_path.write_text(domain_pddl, encoding="utf-8")
                prob_path.write_text(problem_pddl, encoding="utf-8")
                
                reader = self.PDDLReader(raise_on_error=True, strict_with_requirements=False)
                reader.parse_domain(str(dom_path))
                reader.parse_instance(str(prob_path))
                return True
        except Exception:
            return False
    
    def parse_pddl_pair(self, domain_pddl: str, problem_pddl: str, validate: bool = True) -> Dict[str, Any]:
        """
        Parse a domain+problem PDDL pair and return comprehensive parsing results.
        
        Args:
            domain_pddl: PDDL domain string
            problem_pddl: PDDL problem string
            validate: Whether to validate with tarski (if available)
        
        Returns:
            Dictionary containing:
            - domain_name: extracted domain name
            - objects: objects grouped by type
            - initial_state: list of initial state predicates
            - domain_tree: parsed domain S-expression
            - problem_tree: parsed problem S-expression
            - valid: whether tarski validation passed (if attempted)
        """
        # Parse domain
        domain_name, domain_tree = self.parse_pddl_domain(domain_pddl)
        
        # Parse problem
        objects, initial_state, problem_tree = self.parse_pddl_problem(problem_pddl)
        
        # Validate if requested
        valid = True
        if validate:
            valid = self.validate_with_tarski(domain_pddl, problem_pddl)
        
        return {
            "domain_name": domain_name,
            "objects": objects,
            "initial_state": initial_state,
            "domain_tree": domain_tree,
            "problem_tree": problem_tree,
            "valid": valid,
            "tarski_available": self.tarski_available
        }
    
    # ---------------------------
    # Utility Methods
    # ---------------------------
    
    @staticmethod
    def domain_matches(domain_name: str, domain_filter: Optional[str]) -> bool:
        """
        Check if domain name matches the given filter with improved matching logic.
        
        Handles common naming variations:
        - floortile <-> floor-tile
        - grippers <-> gripper-strips  
        - rovers <-> rover
        - visitall <-> grid-visit-all
        """
        if not domain_filter:
            return True
        
        domain_name_lower = (domain_name or "").lower()
        filter_lower = domain_filter.lower()
        
        # Direct substring match
        if filter_lower in domain_name_lower:
            return True
        
        # Handle common domain name variations
        domain_mappings = {
            'floortile': ['floor-tile', 'floortile'],
            'grippers': ['gripper-strips', 'grippers', 'gripper'],
            'rovers': ['rover', 'rovers'],
            'visitall': ['grid-visit-all', 'visitall', 'visit-all'],
            'blocksworld': ['blocksworld', 'blocks-world', 'blockworld'],
            'logistics': ['logistics', 'logistics-strips'],
            # Additional mappings for benchmark variants
            'alfworld': ['alfworld', 'alfred'],
            'goldminer': ['goldminer', 'gold-miner', 'gold-miner-typed'],
            'depot': ['depot','depots'],
            'frogs_jumping': ['frogs_jumping', 'frogs-jumping'],
        }
        
        # Check if filter matches any known variations
        for filter_key, variations in domain_mappings.items():
            if filter_lower == filter_key:
                for variation in variations:
                    if variation in domain_name_lower:
                        return True
        
        # Check reverse mapping (if domain name is a known variation)
        for filter_key, variations in domain_mappings.items():
            if domain_name_lower in [v.lower() for v in variations]:
                if filter_lower == filter_key or filter_lower in [v.lower() for v in variations]:
                    return True
        
        return False
    
    @staticmethod
    def normalize_predicate(pred_name: str, args: List[str]) -> str:
        """Normalize a predicate to standard string format."""
        if not args:
            return pred_name
        return f"({pred_name} {' '.join(args)})"
    
    def format_state(self, predicates: List[Tuple[str, List[str]]]) -> List[str]:
        """Format list of predicates into normalized string representations."""
        return [self.normalize_predicate(pred, args) for pred, args in predicates]


# Global parser instance for easy import
unified_parser = UnifiedPDDLParser()
