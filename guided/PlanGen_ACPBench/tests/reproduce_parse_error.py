
from test_case_generator.unified_pddl_parser import unified_parser

domain_pddl = """(define (domain frogs-jumping)
    (:requirements :strips :typing)
    (:types leftfrog rightfrog - frog frog lilypad - object)
    (:predicates (at ?f - frog ?p - lilypad)  (empty ?p - lilypad)  (next ?p1 - lilypad ?p2 - lilypad))
    (:action jump-left
        :parameters (?rf - rightfrog ?from - lilypad ?mid - lilypad ?to - lilypad ?lf - leftfrog)
        :precondition (and (at ?rf ?from) (next ?to ?mid) (next ?mid ?from) (at ?lf ?mid) (empty ?to))
        :effect (and (not (at ?rf ?from)) (at ?rf ?to) (empty ?from) (not (empty ?to)))
    )
     (:action jump-right
        :parameters (?lf - leftfrog ?from - lilypad ?mid - lilypad ?to - lilypad ?rf - rightfrog)
        :precondition (and (at ?lf ?from) (next ?from ?mid) (next ?mid ?to) (at ?rf ?mid) (empty ?to))
        :effect (and (not (at ?lf ?from)) (at ?lf ?to) (empty ?from) (not (empty ?to)))
    )
     (:action slide-left
        :parameters (?rf - rightfrog ?from - lilypad ?to - lilypad)
        :precondition (and (at ?rf ?from) (next ?to ?from) (empty ?to))
        :effect (and (not (at ?rf ?from)) (at ?rf ?to) (empty ?from) (not (empty ?to)))
    )
     (:action slide-right
        :parameters (?lf - leftfrog ?from - lilypad ?to - lilypad)
        :precondition (and (at ?lf ?from) (next ?from ?to) (empty ?to))
        :effect (and (not (at ?lf ?from)) (at ?lf ?to) (empty ?from) (not (empty ?to)))
    )
)"""

problem_pddl = """(define (problem frogs-3v3)
    (:domain frogs-jumping)
    (:requirements :strips :typing)
    (:objects l1 l2 l3 - leftfrog p1 p2 p3 p4 p5 p6 p7 - lilypad r1 r2 r3 - rightfrog)
    (:init (at l1 p1) (at l2 p5) (at l3 p6) (at r1 p2) (at r2 p3) (at r3 p7) (empty p4) (next p1 p2) (next p2 p3) (next p3 p4) (next p4 p5) (next p5 p6) (next p6 p7))
    (:goal (and (at l3 p1) (at r2 p3)))
)"""

print(f"Tarski available: {unified_parser.tarski_available}")
print("Attempting to parse...")
try:
    result = unified_parser.parse_pddl_pair(domain_pddl, problem_pddl, validate=True)
    print("Parsing successful!")
    print(f"Domain name: {result['domain_name']}")
    print(f"Valid: {result['valid']}")
except Exception as e:
    print(f"Parsing failed: {e}")
