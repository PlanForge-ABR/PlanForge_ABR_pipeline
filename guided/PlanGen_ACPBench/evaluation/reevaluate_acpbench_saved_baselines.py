#!/usr/bin/env python3
"""Re-evaluate saved ACPBench-final CoT/SCOPE plans with VAL; no model rerun."""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scope.results.run_val_acp_bench import verify_plan_with_val


def load_results(pattern):
    output = {}
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8") as file:
            for entry in json.load(file).get("results", []):
                output[entry["id"]] = entry
    return output


def cot_actions(example, result):
    """Keep only CoT lines that begin with a declared PDDL action name."""
    action_names = set(re.findall(r"\(:action\s+([^\s\)]+)", example["PDDL_domain"], re.I))
    action_names = {name.lower() for name in action_names}
    output = []
    for raw in result.get("predicted_plan", []):
        line = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", str(raw).strip())
        match = re.match(r"^\(?\s*([\w-]+)(?:\s|\(|$)", line)
        if match and match.group(1).lower() in action_names:
            output.append(line)
    return output


def val(example, plan):
    if not plan:
        return False, "No parseable action sequence"
    result = verify_plan_with_val(example["PDDL_domain"], example["PDDL_problem"], plan)
    return bool(result["valid"]), result["reason"]


def main():
    output_path = ROOT / "new_results/evaluation_summaries/3_methods_eval_recomputed.json"
    dataset = {}
    for path in (ROOT / "data/ACPBench_dataset_final/test").glob("*.json"):
        with open(path, encoding="utf-8") as file:
            dataset.update({example["id"]: example for example in json.load(file)})
    cot = load_results(str(ROOT / "new_results/cot/*_cot_test_results.json"))
    scope = load_results(str(ROOT / "new_results/scope/*_scope_test_results.json"))
    with open(ROOT / "new_results/evaluation_summaries/3_methods_eval.json", encoding="utf-8") as file:
        planforge = {row["id"]: row.get("planforge_valid") for row in json.load(file)}

    rows = []
    for example_id, example in sorted(dataset.items()):
        if str(example.get("answer")).lower() != "yes":
            continue
        cot_result = cot.get(example_id)
        cot_plan = cot_actions(example, cot_result) if cot_result else []
        cot_valid, cot_reason = val(example, cot_plan) if cot_result else (None, "Missing CoT output")
        scope_result = scope.get(example_id)
        scope_plan = scope_result.get("predicted_plan", []) if scope_result else []
        scope_valid, scope_reason = val(example, scope_plan) if scope_result else (None, "Missing SCOPE output")
        rows.append({
            "id": example_id, "domain": next((p.stem for p in (ROOT / "data/ACPBench_dataset_final/test").glob("*.json") if example_id in {e["id"] for e in json.load(open(p))}), "unknown"),
            "plan_length": example.get("plan_length"), "n_objects": example.get("n_objects"),
            "cot_valid": cot_valid, "cot_action_count": len(cot_plan), "cot_val_reason": cot_reason,
            "scope_valid": scope_valid, "scope_action_count": len(scope_plan) if isinstance(scope_plan, list) else 0, "scope_val_reason": scope_reason,
            "planforge_valid": planforge.get(example_id),
        })

    summary = {}
    for method in ("cot", "scope", "planforge"):
        values = [row[f"{method}_valid"] for row in rows if row[f"{method}_valid"] is not None]
        summary[method] = {"available": len(values), "valid": sum(values), "accuracy": 100 * sum(values) / len(values) if values else None}
    payload = {"metadata": {"metric": "VAL accuracy on ground-truth positive plans", "summary": summary, "scope_note": "32 Rovers positives have no saved SCOPE output and are unavailable, not failures.", "cot_note": "Only recognized PDDL action lines are passed to VAL."}, "instances": rows}
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print("Saved", output_path)
    print(summary)


if __name__ == "__main__":
    main()
