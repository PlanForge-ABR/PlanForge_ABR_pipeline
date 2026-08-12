#!/usr/bin/env python3
"""
End-to-end development driver.

This script automates the full pipeline for all planning domains:

1. Generate / refine `succ.py` and `is_goal.py` via `src/codingagent.py`.
2. Run prompt optimisation for NL→state (`src/nl2state.py` or `src/nl2state_2.py`).
3. Run search over train data (`src/search.py`).
4. Compute per-domain and overall search accuracies (`src/results.py`).

By the end of a successful *dev* run you should have, for each processed domain:
  - `src/<domain>/succ.py`
  - `src/<domain>/is_goal.py`
  - Prompt-optimised module files saved under `src/<domain>/`
  - `src/<domain>/dev_nl2state_result.json`        (NL→state predictions on train)
  - `src/<domain>/dev_search_result.json`          (search results over train)
  - `src/<domain>/dev_search_accuracy.json`        (per-domain metrics)
and one global (dev aggregate):
  - `src/dev_accuracy_summary.json`                (overall metrics)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def run_cmd(args: List[str], cwd: Path) -> int:
    """Run a subprocess, streaming output, and return its exit code."""
    print(f"\n[develop] Running in {cwd}:")
    print("[develop] $ " + " ".join(args))
    try:
        proc = subprocess.run(args, cwd=str(cwd))
        if proc.returncode != 0:
            print(f"[develop] Command failed with exit code {proc.returncode}")
        return proc.returncode
    except FileNotFoundError as exc:
        print(f"[develop] Failed to run {args[0]}: {exc}")
        return 127


def discover_domains(src_dir: Path) -> List[str]:
    """Return a sorted list of domain names under src/."""
    domains: List[str] = []
    for entry in src_dir.iterdir():
        if not entry.is_dir():
            continue
        # Heuristic: treat as a domain if it contains a heuristics.py
        if (entry / "heuristics.py").exists():
            domains.append(entry.name)
    domains.sort()
    return domains


def ensure_succ_and_is_goal(
    repo_root: Path,
    domain: str,
    provider: str,
    model_name: str,
    force: bool,
) -> None:
    """Invoke codingagent.py to (re)generate succ.py and is_goal.py for a domain."""
    src_dir = repo_root / "src"
    dom_dir = src_dir / domain
    succ_py = dom_dir / "succ.py"
    is_goal_py = dom_dir / "is_goal.py"

    already_present = succ_py.exists() and is_goal_py.exists()
    if already_present and not force:
        print(f"[develop] {domain}: succ.py and is_goal.py already exist; skipping CodingAgent (use --force-codingagent to override).")
        return

    goal = f"Implement succ.py and is_goal.py for domain '{domain}' and make python test_{domain}.py pass."
    cmd = [
        sys.executable,
        "src/codingagent.py",
        "--domain",
        domain,
        "--goal",
        goal,
        "--project-root",
        str(dom_dir),
        "--provider",
        provider,
        "--template",
        "src/CodingAgent.md",
    ]
    if model_name:
        cmd.extend(["--model-name", model_name])

    code = run_cmd(cmd, cwd=repo_root)
    if code != 0:
        print(f"[develop] WARNING: CodingAgent run for domain '{domain}' exited with {code}. Continuing.")


def run_nl2state_single_domain(
    repo_root: Path,
    domain: str,
    train_dir: Path,
    n_examples: int,
    n_train: int,
    n_validation: int,
    model: str,
    model_name: str,
    enable_thinking: bool,
) -> None:
    """Run per-domain prompt optimisation via nl2state.py."""
    cmd = [
        sys.executable,
        "src/nl2state.py",
        "--train",
        str(train_dir),
        "--N",
        str(n_examples),
        "--N_train",
        str(n_train),
        "--N_validation",
        str(n_validation),
        "--domain",
        domain,
        "--out",
        "dev_nl2state_result.json",
        "--model",
        model,
        "--model_name",
        model_name,
        "--src",
        "src",
    ]
    if enable_thinking:
        cmd.append("--enable_thinking")

    code = run_cmd(cmd, cwd=repo_root)
    if code != 0:
        print(f"[develop] WARNING: nl2state.py run for domain '{domain}' exited with {code}. Continuing.")


def run_nl2state_multi_domain(
    repo_root: Path,
    domains: Iterable[str],
    train_dir: Path,
    n_examples: int,
    n_train_per_domain: int,
    n_validation: int,
    model: str,
    model_name: str,
    enable_thinking: bool,
) -> None:
    """
    Run multi-domain prompt optimisation via nl2state_2.py.

    The first domain call will train and save a shared multi-domain module
    under src/, subsequent domains will reuse it via --load_module.
    Domain-specific nl2state_result.json files are written into each domain.
    """
    repo_root = repo_root.resolve()
    src_dir = repo_root / "src"
    module_file = "multi_domain_optimized_module.json"
    module_path = src_dir / module_file

    domains = list(domains)
    if not domains:
        return

    # First domain: train multi-domain module (unless it already exists).
    first = domains[0]
    if module_path.exists():
        print(f"[develop] Multi-domain module already exists at {module_path}; reusing for all domains.")
        first = None

    for idx, domain in enumerate(domains):
        use_load_module = module_path.exists() or (idx > 0 and first is not None)

        cmd = [
            sys.executable,
            "src/nl2state_2.py",
            "--train",
            str(train_dir),
            "--N",
            str(n_examples),
            "--N_train",
            str(n_train_per_domain),
            "--N_validation",
            str(n_validation),
            "--domain",
            domain,
            "--out",
            "dev_nl2state_result.json",
            "--model",
            model,
            "--model_name",
            model_name,
            "--src",
            "src",
        ]
        if enable_thinking:
            cmd.append("--enable_thinking")
        # Use all discovered domains for cross-domain optimisation.
        cmd.extend(["--domains", *domains])

        if use_load_module and module_path.exists():
            cmd.extend(["--load_module", module_file])

        code = run_cmd(cmd, cwd=repo_root)
        if code != 0:
            print(f"[develop] WARNING: nl2state_2.py run for domain '{domain}' exited with {code}. Continuing.")

    # Optionally copy the shared multi-domain module into each domain folder
    # so every domain has a local copy of the prompt-optimised module.
    if module_path.exists():
        prompt_path = src_dir / module_path.name.replace(".json", "_prompt.txt")
        for domain in domains:
            dom_dir = src_dir / domain
            dom_dir.mkdir(parents=True, exist_ok=True)
            dest_module = dom_dir / module_path.name
            print(f"[develop] Copying multi-domain module to {dest_module}")
            dest_module.write_bytes(module_path.read_bytes())
            if prompt_path.exists():
                dest_prompt = dom_dir / prompt_path.name
                print(f"[develop] Copying multi-domain prompt to {dest_prompt}")
                dest_prompt.write_bytes(prompt_path.read_bytes())


def run_search_for_domain(
    repo_root: Path,
    domain: str,
    timeout: int,
    strategy: str,
) -> None:
    """Run search.py for a given domain using its nl2state_result.json (train)."""
    cmd = [
        sys.executable,
        "src/search.py",
        "--domain",
        domain,
        "--timeout",
        str(timeout),
        "--strategy",
        strategy,
        "--src",
        "src",
        "--nl2state",
        "dev_nl2state_result.json",
        "--out",
        "dev_search_result.json",
    ]
    code = run_cmd(cmd, cwd=repo_root)
    if code != 0:
        print(f"[develop] WARNING: search.py run for domain '{domain}' exited with {code}. Continuing.")


def run_results(
    repo_root: Path,
    split: str,
) -> None:
    """Run results.py to compute per-domain and overall accuracy."""
    cmd = [
        sys.executable,
        "src/results.py",
        "--split",
        split,
        "--out",
        "dev_search_result.json",
        "--dom_summary_out",
        "dev_search_accuracy.json",
        "--summary_out",
        "dev_accuracy_summary.json",
    ]
    code = run_cmd(cmd, cwd=repo_root)
    if code != 0:
        print(f"[develop] WARNING: results.py exited with {code}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: CodingAgent → prompt optimisation → search → accuracy."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domains to process (default: all src/* domains containing heuristics.py).",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "codex", "qwen"],
        default="codex",
        help="CLI coding agent provider used by codingagent.py.",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Optional model name passed through to codingagent.py (e.g., gemini-2.5-flash).",
    )
    parser.add_argument(
        "--force-codingagent",
        action="store_true",
        help="Always run CodingAgent even if succ.py and is_goal.py already exist.",
    )
    parser.add_argument(
        "--skip-codingagent",
        action="store_true",
        help="Skip the CodingAgent succ/is_goal generation step.",
    )
    parser.add_argument(
        "--multi-domain-prompt",
        action="store_true",
        help="Use multi-domain prompt optimisation via nl2state_2.py instead of per-domain nl2state.py.",
    )
    parser.add_argument(
        "--train-dir",
        default="data/train",
        help="Path to training data directory (as expected by nl2state scripts).",
    )
    parser.add_argument(
        "--nl2state-N",
        type=int,
        default=20,
        help="Number of training examples per domain for NL→state processing.",
    )
    parser.add_argument(
        "--nl2state-N-train",
        type=int,
        default=4,
        help="Number of examples for prompt optimisation (per-domain or per-domain in multi-domain mode).",
    )
    parser.add_argument(
        "--nl2state-N-validation",
        type=int,
        default=10,
        help="Number of validation examples for NL→state prompt optimisation.",
    )
    parser.add_argument(
        "--nl2state-model",
        default="openai",
        choices=["openai", "gemini"],
        help="Model provider for nl2state scripts.",
    )
    parser.add_argument(
        "--nl2state-model-name",
        default="gpt-5.1",
        help="Model name for nl2state scripts (e.g., gpt-5.1, gemini-2.5-flash).",
    )
    parser.add_argument(
        "--nl2state-enable-thinking",
        action="store_true",
        help="Enable DSPy thinking mode in nl2state scripts.",
    )
    parser.add_argument(
        "--search-timeout",
        type=int,
        default=20,
        help="Search timeout per example in seconds.",
    )
    parser.add_argument(
        "--search-strategy",
        default="astar",
        choices=["bfs", "a*", "astar", "a-star"],
        help="Search strategy for search.py.",
    )
    parser.add_argument(
        "--results-split",
        default="train",
        choices=["train", "test", "none"],
        help="Which split to use for ground-truth answers in results.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"

    if not src_dir.is_dir():
        print(f"[develop] src directory not found at {src_dir}")
        return 1

    if args.domains:
        domains = sorted(args.domains)
    else:
        domains = discover_domains(src_dir)

    if not domains:
        print("[develop] No domains discovered; nothing to do.")
        return 1

    print(f"[develop] Domains to process: {', '.join(domains)}")

    # 1) succ.py / is_goal.py via CodingAgent
    if not args.skip_codingagent:
        for domain in domains:
            ensure_succ_and_is_goal(
                repo_root=repo_root,
                domain=domain,
                provider=args.provider,
                model_name=args.model_name,
                force=args.force_codingagent,
            )
    else:
        print("[develop] Skipping CodingAgent succ/is_goal generation step.")

    # 2) Prompt optimisation and NL→state processing
    train_dir = (repo_root / args.train_dir).resolve()
    if not train_dir.exists():
        print(f"[develop] Training data directory not found: {train_dir}")
        return 1

    if args.multi_domain_prompt:
        print("[develop] Using multi-domain prompt optimisation (nl2state_2.py).")
        run_nl2state_multi_domain(
            repo_root=repo_root,
            domains=domains,
            train_dir=train_dir,
            n_examples=args.nl2state_N,
            n_train_per_domain=args.nl2state_N_train,
            n_validation=args.nl2state_N_validation,
            model=args.nl2state_model,
            model_name=args.nl2state_model_name,
            enable_thinking=args.nl2state_enable_thinking,
        )
    else:
        print("[develop] Using per-domain prompt optimisation (nl2state.py).")
        for domain in domains:
            run_nl2state_single_domain(
                repo_root=repo_root,
                domain=domain,
                train_dir=train_dir,
                n_examples=args.nl2state_N,
                n_train=args.nl2state_N_train,
                n_validation=args.nl2state_N_validation,
                model=args.nl2state_model,
                model_name=args.nl2state_model_name,
                enable_thinking=args.nl2state_enable_thinking,
            )

    # 3) Search over train data for each domain
    for domain in domains:
        run_search_for_domain(
            repo_root=repo_root,
            domain=domain,
            timeout=args.search_timeout,
            strategy=args.search_strategy,
        )

    # 4) Aggregate search accuracies
    run_results(
        repo_root=repo_root,
        split=args.results_split,
    )

    print("\n[develop] Pipeline completed. Check per-domain folders under src/ and src/dev_accuracy_summary.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
