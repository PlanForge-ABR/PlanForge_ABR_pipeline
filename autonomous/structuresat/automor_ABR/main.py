"""PlanForge ABR entry point for the automor SAT domain."""

from __future__ import annotations

import json
from pathlib import Path

from runner.pipeline import run_planforge


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset_root = root.parent / "automor_balanced"
    output_dir = root / "outputs"

    summary = run_planforge(dataset_root=dataset_root, output_dir=output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
