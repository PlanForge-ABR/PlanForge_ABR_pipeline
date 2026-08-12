"""Execution, validation, freeze, and evaluation pipeline for PS SAT."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from architect.specification import FROZEN_SPEC
from builder.cnf import assignment_satisfies, parse_dimacs
from builder.solver import solve


@dataclass(frozen=True)
class InstanceRecord:
    path: Path
    label: str


@dataclass
class PhaseMetrics:
    phase: str
    total: int
    successes: int
    correct_predictions: int
    total_runtime_seconds: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.total if self.total else 0.0

    @property
    def average_runtime_seconds(self) -> float:
        return self.total_runtime_seconds / self.total if self.total else 0.0


def discover_instances(dataset_root: Path) -> list[InstanceRecord]:
    train_roots = sorted(path for path in dataset_root.rglob("train") if path.is_dir())
    if not train_roots:
        raise FileNotFoundError(f"No train directory found under dataset root: {dataset_root}")

    records: list[InstanceRecord] = []
    for train_root in train_roots:
        for path in sorted(train_root.rglob("*.cnf")):
            label = path.parent.name.upper()
            if label not in {"SAT", "UNSAT"}:
                raise ValueError(f"CNF file must be under sat/ or unsat/ label folder: {path}")
            records.append(InstanceRecord(path=path, label=label))

    records = sorted(records, key=lambda record: str(record.path))
    if len(records) < 1020:
        raise ValueError(f"Need at least 1020 instances, found {len(records)} in {dataset_root}")
    return records


def run_planforge(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_instances = discover_instances(dataset_root)
    development = all_instances[:20]
    evaluation = all_instances[20:1020]

    loop_log: list[dict[str, object]] = [
        {
            "attempt": 1,
            "architect_methodology": asdict(FROZEN_SPEC),
            "builder_implementation": "builder.cnf + builder.solver",
            "runner_scope": "first 20 discovered instances",
        }
    ]

    dev_results, dev_metrics = run_phase("development", development)
    loop_log[0]["runner_feedback"] = metrics_dict(dev_metrics)
    loop_log[0]["passed"] = dev_metrics.successes == dev_metrics.total
    write_json(output_dir / "development_results.json", dev_results)
    write_json(output_dir / "development_metrics.json", metrics_dict(dev_metrics))
    write_json(output_dir / "development_loop.json", loop_log)

    if dev_metrics.successes != dev_metrics.total:
        failures = [row for row in dev_results if row["status"] != "SUCCESS"]
        write_json(output_dir / "development_failures.json", failures)
        raise RuntimeError(
            "Development phase did not reach 100% success; evaluation is blocked by freeze rule."
        )

    freeze_metadata = {
        "frozen": True,
        "freeze_reason": "All 20 development instances returned SUCCESS.",
        "architect_specification": asdict(FROZEN_SPEC),
        "development_metrics": metrics_dict(dev_metrics),
        "evaluation_instances": len(evaluation),
    }
    write_json(output_dir / "freeze_metadata.json", freeze_metadata)

    eval_results, eval_metrics = run_phase("evaluation", evaluation)
    write_json(output_dir / "evaluation_results.json", eval_results)
    write_json(output_dir / "evaluation_metrics.json", metrics_dict(eval_metrics))

    summary = {
        "architect_specification": asdict(FROZEN_SPEC),
        "development": metrics_dict(dev_metrics),
        "evaluation": metrics_dict(eval_metrics),
        "outputs": {
            "development_results": str(output_dir / "development_results.json"),
            "development_metrics": str(output_dir / "development_metrics.json"),
            "development_loop": str(output_dir / "development_loop.json"),
            "freeze_metadata": str(output_dir / "freeze_metadata.json"),
            "evaluation_results": str(output_dir / "evaluation_results.json"),
            "evaluation_metrics": str(output_dir / "evaluation_metrics.json"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_phase(
    phase: str,
    instances: list[InstanceRecord],
) -> tuple[list[dict[str, object]], PhaseMetrics]:
    phase_start = perf_counter()
    rows: list[dict[str, object]] = []
    successes = 0
    correct = 0
    solver_runtime = 0.0

    for index, instance in enumerate(instances, start=1):
        row = run_instance(phase, index, instance)
        rows.append(row)
        if row["status"] == "SUCCESS":
            successes += 1
        if row["prediction"] == instance.label:
            correct += 1
        solver_runtime += float(row["runtime_seconds"])

    elapsed = perf_counter() - phase_start
    metrics = PhaseMetrics(
        phase=phase,
        total=len(instances),
        successes=successes,
        correct_predictions=correct,
        total_runtime_seconds=max(solver_runtime, elapsed),
    )
    return rows, metrics


def run_instance(phase: str, index: int, instance: InstanceRecord) -> dict[str, object]:
    try:
        formula = parse_dimacs(instance.path)
        result = solve(formula)
        status = result.status

        if result.prediction == "SAT":
            valid_assignment = (
                result.assignment is not None
                and assignment_satisfies(formula.clauses, result.assignment)
            )
            if not valid_assignment:
                status = "FAILURE"

        return {
            "phase": phase,
            "index": index,
            "instance": str(instance.path),
            "label": instance.label,
            "status": status,
            "prediction": result.prediction,
            "assignment": stringify_assignment(result.assignment),
            "correct_prediction": result.prediction == instance.label,
            "runtime_seconds": result.runtime_seconds,
            "decisions": result.decisions,
            "error": None,
        }
    except Exception as exc:
        return {
            "phase": phase,
            "index": index,
            "instance": str(instance.path),
            "label": instance.label,
            "status": "FAILURE",
            "prediction": None,
            "assignment": None,
            "correct_prediction": False,
            "runtime_seconds": 0.0,
            "decisions": 0,
            "error": str(exc),
        }


def stringify_assignment(assignment: dict[int, bool] | None) -> dict[str, bool] | None:
    if assignment is None:
        return None
    return {str(variable): assignment[variable] for variable in sorted(assignment)}


def metrics_dict(metrics: PhaseMetrics) -> dict[str, object]:
    return {
        "phase": metrics.phase,
        "total": metrics.total,
        "successes": metrics.successes,
        "correct_predictions": metrics.correct_predictions,
        "success_rate": metrics.success_rate,
        "sat_unsat_accuracy": metrics.accuracy,
        "total_runtime_seconds": metrics.total_runtime_seconds,
        "average_runtime_seconds": metrics.average_runtime_seconds,
    }


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

