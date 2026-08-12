#!/usr/bin/env python3
"""
codingagent.py — minimal command-only launcher (single provider, no local loops)

This script ONLY:
  • fills a Markdown prompt template (CodingAgent.md or --template you pass),
  • sends the filled prompt to ONE chosen CLI provider (gemini / codex / qwen),
  • exits with the provider's return code.

It does NOT:
  • run tests,
  • parse provider output,
  • apply file edits,
  • create state directories/files,
  • iterate over providers or limit steps.

All execution (edit → run `python test_{DOMAIN}.py` → repeat) is the CLI agent's job.

Example:
  python src/codingagent.py \
    --domain alfworld \
    --goal "Fix successor & goal until all tests pass" \
    --project-root ./src/alfworld \
    --provider codex \
    --template src/CodingAgent.md
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from src.utils import DOMAIN_DESCRIPTIONS


def which(cmd: str) -> Optional[str]:
    """Return full path if command exists on PATH, else None."""
    return shutil.which(cmd)


def read_text(path: Path) -> str:
    """Read UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def fill_template(tpl: str, mapping: Dict[str, str]) -> str:
    """Very simple {KEY} placeholder replacement."""
    out = tpl
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


# Fallback prompt if --template file not found.
DEFAULT_TEMPLATE = """# CodingAgent — Execution Prompt

- DOMAIN: {DOMAIN}
- DOMAIN DESCRIPTION: {DOMAIN_DESCRIPTION}
- REPO ROOT: {REPO_ROOT}
- GOAL: {GOAL}
- TEST COMMAND: python test_{DOMAIN}.py
- STATE (is_goal): {IS_GOAL_PATH}
- STATE (succ): {SUCC_PATH}
 - MODEL NAME: {MODEL_NAME}

## Role
You are a CLI coding agent with permission to run shell commands and modify files in {REPO_ROOT}.
Your mission is to make minimal, correct changes until running `python test_{DOMAIN}.py` ends with **ALL TESTS PASSED**.

## Hard Rules
1) Never read/import/parse or depend on any test sources or domain test JSONs in your implementation:
   - `test_*.py`
   - `goal_tests.json`
   - `succ_tests.json`
   - any `*_tests.json`
   You MAY execute the tests via shell to observe results.
2) Keep changes surgical and idiomatic. Do not over-refactor.
3) Do not touch any of the forbidden artifacts above.

## What to do (you control the loop)
- Edit code → run `python test_{DOMAIN}.py` → inspect output → repeat until **ALL TESTS PASSED**.
- When done, write:
  - `{IS_GOAL_PATH}` with JSON at least: `{"is_goal": true, "ts": <unix_time>}`
  - `{SUCC_PATH}`   with JSON at least: `{"succ": true, "ts": <unix_time>}`
- If you give up, write both with `false` and a short reason in `"notes"`.
"""


# provider_name: (mode, base_cmd)
# mode "stdin": pass prompt via STDIN
# mode "arg":   append prompt as a single CLI argument
PROVIDERS: Dict[str, Tuple[str, List[str]]] = {
    "gemini": ("arg",  ["gemini", "-y", "-p"]),
    "codex":  ("arg",  ["codex", "exec", "--full-auto", "-s", "workspace-write"]),
    "qwen":   ("stdin",["qwen", "-y", "-p"]),
}


def parse_args(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(
        description="Send a filled CodingAgent prompt to ONE CLI coding agent."
    )
    ap.add_argument(
        "--domain", required=True,
        help="Domain name (used to form test filename and state paths)"
    )
    ap.add_argument(
        "--goal", required=True,
        help="High-level objective for this run"
    )
    ap.add_argument(
        "--project-root", default=".",
        help="Path to the repository root (working directory for the CLI agent)"
    )
    ap.add_argument(
        "--provider", choices=list(PROVIDERS.keys()), default="gemini",
        help="Which single provider to run"
    )
    ap.add_argument(
        "-m", "--model-name", dest="model_name", default="",
        help="Optional model name to include in the prompt (for example: gemini-2.5-flash)"
    )
    ap.add_argument(
        "--template", default="CodingAgent.md",
        help="Markdown template to fill and send"
    )
    ap.add_argument(
        "--state-dir", default=".agent_state",
        help="Where the agent should write state JSONs (it will create the dirs)"
    )
    ap.add_argument(
        "--provider-cmd", nargs="+",
        help="Override the provider command entirely (prompt will be piped via STDIN)"
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    project_root = Path(args.project_root).resolve()
    # The launcher does not create these paths; it only passes them in the prompt.
    is_goal_path = str(project_root / args.state_dir / args.domain / "is_goal.json")
    succ_path = str(project_root / args.state_dir / args.domain / "succ.json")

    # Load template (or fallback)
    tpl_path = Path(args.template)
    if tpl_path.exists():
        print("Successfully Read template")
        template_text = read_text(tpl_path)
    else:
        print("Error Reading Template")
        template_text = DEFAULT_TEMPLATE

    # Fill placeholders
    prompt = fill_template(
        template_text,
        {
            "DOMAIN": args.domain,
            "DOMAIN_DESCRIPTION": DOMAIN_DESCRIPTIONS.get(args.domain, "No description available."),
            "REPO_ROOT": str(project_root),
            "GOAL": args.goal,
            "IS_GOAL_PATH": is_goal_path,
            "SUCC_PATH": succ_path,
        },
    )

    # Resolve provider command
    if args.provider_cmd:
        mode, cmd = "stdin", args.provider_cmd
    else:
        mode, cmd = PROVIDERS[args.provider]
        # If user provided a model name, insert it right after the provider
        # executable (e.g. `gemini <model>`). Make a copy to avoid mutating
        # the global PROVIDERS table.
        if args.model_name:
            cmd = list(cmd)  # shallow copy
            # Insert flag and model name immediately after the executable.
            cmd[1:1] = ["-m", args.model_name]
        # Basic availability check for the base executable
        exe = cmd[0]
        if which(exe) is None:
            print(f"[codingagent] provider CLI '{exe}' not found on PATH", file=sys.stderr)
            return 127

    # Launch from repo root; provider handles everything else
    if mode == "stdin":
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            cwd=str(project_root),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return proc.returncode
    else:  # mode == "arg"
        proc = subprocess.run(
            cmd + [prompt],
            cwd=str(project_root),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
