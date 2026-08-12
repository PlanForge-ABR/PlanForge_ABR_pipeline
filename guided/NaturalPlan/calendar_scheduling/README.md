# PlanForge-Guided Calendar Scheduling

This folder implements the methodology in `../AB_methodology.md` for
`../NaturalPlan/data/calendar_scheduling.json`.

Run the development slice:

```powershell
python main.py --limit 20
```

Run the full evaluation:

```powershell
python main.py
```

Outputs are written to `outputs/calendar_predictions.csv`. Metrics:

- `SR`: fraction of predicted plans satisfying parsed calendar constraints.
- `EM`: exact normalized match with `golden_plan`.

