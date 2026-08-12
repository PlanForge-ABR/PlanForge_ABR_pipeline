# Meeting Planning ABR Solver

This folder contains an executable Architect-Builder-Runner implementation for
`NaturalPlan/data/meeting_planning.json`.

## Files

- `meeting_planner.py`: architect specification, parser, builder search logic,
  runner entrypoint, plan formatter, and internal verifier.
- `evaluate_meeting_planning.py`: standalone evaluator for Success Rate (SR)
  and Exact Match (EM).
- `meeting_planning_eval_results.json`: detailed full-dataset results generated
  by the evaluator.

## Method

The architect represents each instance as a time-window routing problem:

- state: current location, current time, and already-met friends
- action: travel to a feasible friend and meet for the required duration
- objective: maximize the number of friends met
- tie handling: deterministic, preferring earlier completion and stable input
  order

The builder implements a deterministic dynamic search over feasible meeting
orders. The runner calls the builder and returns:

```json
{"status": "SUCCESS", "plan": ["..."]}
```

The evaluator verifies generated schedules against travel times, availability
windows, and meeting durations, then compares the generated plan to
`golden_plan` only for EM calculation.

## Run

Development set, first 20 instances:

```powershell
python .\evaluate_meeting_planning.py --limit 20 --no-output
```

Full dataset:

```powershell
python .\evaluate_meeting_planning.py
```

The full run writes `meeting_planning_eval_results.json` by default.

## Current Results

On all 1000 instances:

- SR: `1.0`
- EM: `0.404`

On the first 20 development instances:

- SR: `1.0`
- EM: `1.0`
