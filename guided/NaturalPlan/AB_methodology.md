# PlanForge-guided methodology for solving planning problems for Natural Plan dataset.

## 🔧 OVERALL SYSTEM (MENTAL MODEL)

You are still doing PlanForge—but reorganized:

- Architect = defines the planning framework (static, reusable)
- Builder = generates problem-specific logic (LLM agent)
- Runner = executes search + evaluates

So your pipeline becomes:
`Natural Language → (Builder) → State + Functions → (Runner) → Search → Plan / No Plan`

This still aligns perfectly with PlanForge’s induced model idea

## STEP 1 — ARCHITECT (YOU DEFINE THE RULES) -> This is static code (no LLM here)

### 📄 `state_schema.py`

Define Trip Planning State Representation

What goes inside:

- Cities visited so far
- Current city
- Remaining cities
- Day index
- Constraints (if any)

Example (conceptually):
```
State:
    current_city
    visited_cities
    remaining_cities
    total_days_used
```
👉 Output of this file: A strict schema the Builder MUST follow

### 📄 `abstract_planner.py`

Define abstract methods (interfaces only):

- get_initial_state()
- get_goal_test(state)
- get_successors(state)

Initial state: 
1. In trip_planning dataset, initial state he have'nt visited any city.
2. In Meeting_Planning dataset, initial state is the first constraint given in the constraints key of json files.
3. In calender_scheduling dataset, initial state is empty means the tiem slot checking will start from the working hours of that day.

No implementation, just signatures.

👉 This is what Builder must implement.

### 📄 search_library.py

Provide ready-made search algorithms:

- BFS
- DFS
- A* (optional)

Each expects:
```
initial_state
successor_fn
goal_test_fn
```
👉 Important: Do NOT generate these via LLM. Keep deterministic.

### 📄 integration_contract.py

This is critical.

Define:

- Expected input/output formats between:
    - Builder → Runner
    - Runner → Search

Example:
```
Builder must return:
{
    "initial_state": ...,
    "goal_test": function,
    "successor_fn": function
}
```

## STEP 2 — BUILDER (LLM / CODING AGENT PART) -> This is where your PlanForge intelligence lives.

### 📄 nl2state_agent.py

Input: prompt_0shot from dataset

Output:

- Parsed structured data:
    - cities
    - durations
    - constraints

👉 This replaces PlanForge NL2State

You MUST enforce:

- schema validation
- consistency checks

### 📄 function_generator.py

This generates:

1. Initial State
- start city
- remaining cities list
2. Goal Test
- all cities visited?
- total duration matches?
3. Successor Function

Defines valid transitions:
```
From city A → go to city B
IF B not visited
```

### 📄 test_generator.py

Create unit tests automatically

Examples:

- Cannot visit same city twice
- Must visit exactly N cities
- Duration constraints respected

👉 This is your PlanForge test grounding loop

### 📄 refinement_loop.py

This is your core PlanForge loop:
```
repeat:
    generate functions
    run tests
    collect errors
    fix functions
until pass or budget exhausted
```
👉 This ensures correctness BEFORE search

## STEP 3 — RUNNER (EXECUTION PIPELINE)

### 📄 executor.py

Takes Builder output and:

- Loads:
    - initial_state
    - successor_fn
    - goal_test

### 📄 search_runner.py

Runs:
```
plan = search_algorithm(
    initial_state,
    successor_fn,
    goal_test
)
```
If:

- plan found → return plan
- exhausted → return ⊥

👉 This gives Plan Existence

### 📄 evaluation.py

Compare:

- Generated plan vs golden_plan

Metrics:

- Exact match
- Constraint satisfaction
- Plan validity (execution check)

## 🔁 STEP 4 — MAIN PIPELINE

### 📄 main.py

Flow:
```
for each example in trip_planning:

    # BUILDER
    structured_data = nl2state_agent()
    functions = refinement_loop()

    # RUNNER
    result = search_runner()

    # EVALUATION
    evaluate(result, golden_plan)

    save outputs
```

## 🧪 STEP 5 — HOW PLAN EXISTENCE WORKS

You don’t need extra logic.

Search naturally gives:

- ✅ Found → Plan exists
- ❌ Exhausted → No plan

That’s your ⊥ (no-plan) from PlanForge

## 🧭 STEP 6 — HOW PLAN GENERATION WORKS

Search returns:
```
[a1, a2, a3, ...]
```
Convert to:
```
Day 1: City A
Day 2: City B
...
```

## 🔄 STEP 7 — GENERALIZE TO OTHER DATASETS

Once Trip Planning works:

#### Meeting Planning
- State = current location + visited people
- Successor = travel + meeting
- Constraints = time + distance matrix

#### Calendar Scheduling
- State = assigned meetings per day
- Successor = assign meeting slot
- Goal = all meetings scheduled

👉 Only Builder changes
Architect + Runner stay SAME

### Rough idea on how the code base is
```
Builder_architect_planning/
│
├── architect/
│   ├── state_schema.py
│   ├── search_library.py
│   ├── abstract_planner.py
│   └── integration_contract.py
│
├── builder/
│   ├── nl2state_agent.py
│   ├── function_generator.py
│   ├── test_generator.py
│   └── refinement_loop.py
│
├── runner/
│   ├── executor.py
│   ├── search_runner.py
│   └── evaluation.py
│
├── datasets/
│   └── trip_planning.json
│
├── outputs/
│
└── main.py
```

Dataset for Natural_plan is in here : "NaturalPlan\data"
1. NaturalPlan\data\trip_planning.json
2. NaturalPlan\data\meeting_planning.json
3. NaturalPlan\data\calendar_scheduling.json