# Calendar Scheduling ABR Solver

This folder contains an executable Architect-Builder-Runner implementation for
NaturalPlan calendar-scheduling instances.

Run the development split, defined as the first 20 instances:

```powershell
python evaluate_calendar_scheduling.py --limit 20 --no-output
```

Run the full dataset and save predictions plus metric details:

```powershell
python evaluate_calendar_scheduling.py --output calendar_scheduling_eval_results.json
```

Metrics:

- `success_rate`: fraction of generated plans that satisfy all parsed calendar
  constraints.
- `exact_match_rate`: fraction of generated plans that exactly match
  `golden_plan` after whitespace normalization.

The planner does not read `golden_plan` while generating plans. Golden plans are
used only by `evaluate_calendar_scheduling.py` for Exact Match calculation.
