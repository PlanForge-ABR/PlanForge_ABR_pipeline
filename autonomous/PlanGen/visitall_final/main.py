"""Entry point for the visitall_final ABR pipeline."""

import argparse
import json
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from runner.runner import run_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the visitall ABR solver.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--no-write", action="store_true", help="Do not write outputs JSON.")
    args = parser.parse_args()

    summary = run_dataset(split=args.split, write_outputs=not args.no_write)
    public_summary = {
        "domain": summary["domain"],
        "split": summary["split"],
        "instances": summary["instances"],
        "results": summary["results"],
    }
    if args.split == "train":
        public_summary["plan_existence_accuracy"] = summary["plan_existence_accuracy"]
        public_summary["all_train_existence_correct"] = summary["all_train_existence_correct"]
        public_summary["all_train_plans_valid"] = summary["all_train_plans_valid"]

    print(json.dumps(public_summary, indent=2))


if __name__ == "__main__":
    main()
