# Trip Planning ABR Solver

This folder contains an executable Architect-Builder-Runner implementation for
NaturalPlan trip-planning instances.

Run the development split, defined as the first 20 instances:

```powershell
python evaluate_trip_planning.py --limit 20 --no-output
```

Run the full dataset and save predictions plus metric details:

```powershell
python evaluate_trip_planning.py --output trip_planning_eval_results.json
```

Metrics:

- `success_rate`: fraction of generated plans that satisfy all parsed problem constraints.
- `exact_match_rate`: fraction of generated plans that exactly match `golden_plan` after
  trimming trailing whitespace.

The planner does not read `golden_plan` while generating plans. Golden plans are used
only by `evaluate_trip_planning.py` for Exact Match calculation.
