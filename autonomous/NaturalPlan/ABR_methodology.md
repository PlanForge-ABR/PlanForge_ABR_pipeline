# Autonomous Planning via Architect–Builder–Runner (ABR)

## Core Idea

In this methodology,

We allow an LLM-based agent to:
- Design the planning abstraction
- Choose how to represent the problem
- Decide how planning should be solved

The system only enforces one requirement:

> The final solution must be **executable and verifiable**

---

## System Overview

The system consists of three roles:

### 1. Architect (LLM Agent)

The Architect is responsible for designing a planning system for a given natural language problem.

It decides:

- How to represent the problem (state, variables, structures, or any abstraction)
- What functions are required (e.g., transitions, scoring, constraints, or alternatives)
- What solving strategy to use (search, simulation, optimization, heuristic reasoning, etc.)
- How all components interact to produce a plan

The Architect outputs a **complete specification** of a planning system.

---

### 2. Builder (LLM Coding Agent)

The Builder implements the system designed by the Architect.

It:
- Translates natural language input into structured representations (if required)
- Implements all functions defined by the Architect
- Generates executable code for the planning system

The Builder is NOT restricted to:
- Classical planning
- Explicit state transitions
- Goal predicates

It can implement any computational mechanism that produces a plan.

---

### 3. Runner (Execution Engine)

The Runner executes the system produced by the Builder.

It:
- Loads the generated code
- Runs the planning procedure
- Produces:
  - A candidate plan, or
  - A failure signal (no plan found)

The Runner treats the system as a black box.

---

## Key Constraint: Executability and Verification

Even though the Architect has full freedom, the system must satisfy:

1. **Executability**
   - All generated components must run without errors

2. **Verifiability**
   - The produced plan must be checkable against the problem constraints

3. **Deterministic Outcome**
   - Running the same system should produce consistent results

---

## Planning Workflow

For each input example:

### Step 1: Architect Phase

Input:
- Natural language problem (from dataset)

Output:
- A structured system design describing:
  - Representation
  - Functions
  - Solving strategy
  - Integration logic

---

### Step 2: Builder Phase

Input:
- Architect specification
- Natural language problem

Process:
- Generate code implementing the system
- Optionally create internal validation checks

Output:
- Executable planning system

---

### Step 3: Execution Phase (Runner)

- Execute the generated system
- Obtain:
  - Plan (sequence / structure)
  - OR failure

---

### Step 4: Verification Phase

- Check whether:
  - Plan satisfies constraints
  - Plan matches required structure
  - Plan is logically consistent

If verification fails:
- Return failure OR trigger regeneration

---

## Output Definition

For each instance, the system must return:
{
"status": "SUCCESS" | "FAILURE",
"plan": <generated_plan_or_null>
}


---

## Plan Existence

- If a valid plan is produced → SUCCESS
- If no valid plan is found → FAILURE

No separate logic is required.

---

## Dataset Usage

This framework is applied to:

1. Trip Planning
2. Meeting Planning
3. Calendar Scheduling

Each instance provides (only from "prompt_0shot" itself):
- Natural language prompt
- Constraints
- Ground truth plan (mentioned as "golden_plan" key for evaluation only)

The system must operate **without relying on ground truth during generation**.

---

## Evaluation

After execution:

- Compare generated plan with ground truth ("golden_plan")
- Check:
  - Validity
  - Constraint satisfaction
  - Structural correctness

Metrics:
- Exact match accuracy
- Valid plan rate
- Failure detection accuracy

---

## Important Notes

- Do NOT hardcode planning logic
- Do NOT assume classical planning
- Do NOT enforce state-transition structure
- Allow the agent to invent representations and strategies

However:

- Ensure outputs are executable
- Ensure results are verifiable

---

## Expected Behavior of the Agent

The agent should:

- Explore multiple possible representations
- Choose efficient solving strategies
- Adapt methods across different problem types
- Improve through iterative refinement if needed

---

## Summary

This methodology transforms planning from:

"LLM generates plans"

to:

"LLM designs and builds a system that generates plans"

The correctness of the solution is ensured not by the LLM's text generation,
but by executing and verifying the system it creates.