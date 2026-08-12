"""
Zero-Shot DSPy Module (Simplified) - Context and Question only.

This module contains the DSPy module and evaluation loop for zero-shot reasoning
using only context and question/inputs, without domain descriptions or action lists.
"""

import glob
import argparse
import json
import os
from typing import Any, Dict, List, Optional
import dotenv
import dspy

dotenv.load_dotenv()

# ============================================================================
# DSPY SIGNATURE AND MODULE
# ============================================================================

class ZeroShotSignature(dspy.Signature):
    """
    Given a problem context and a question, determine the answer.
    """
    context = dspy.InputField(desc="The problem setup describing objects and relations.")
    question = dspy.InputField(desc="The question to answer about reachability or feasibility.")

    reasoning = dspy.OutputField(desc="Step-by-step reasoning about the problem.")
    final_answer = dspy.OutputField(desc="'yes' or 'no'.")


class ZeroShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(ZeroShotSignature)

    def forward(self, context: str, question: str):
        return self.prog(
            context=context,
            question=question
        )


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model(model_type: str = "openai", model_name: str = "gpt-5.1") -> bool:
    """Setup DSPy language model configuration
    
    Returns:
        bool: True if DSPy thinking should be enabled, False if disabled (for thinking models)
    """
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if model_type.lower() == "openai":
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        lm = dspy.LM(f'openai/{model_name}', api_key=openai_api_key, temperature=1.0, max_tokens=64000, reasoning_effort="medium", cache=False)
        print(f"✓ Configured GPT-5.1 model ({model_name})")
        has_built_in_thinking = True  # GPT-5.1 does have built-in thinking mode

    elif model_type.lower() == "gemini":
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        lm = dspy.LM(f'gemini/{model_name}', api_key=gemini_api_key, max_tokens=32000, cache=False)
        print(f"✓ Configured Gemini model ({model_name})")
        has_built_in_thinking = False
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'openai' or 'gemini'")
    
    dspy.configure(lm=lm)
    return not has_built_in_thinking


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def run_evaluation(
    data_dir: str,
    output_path: str,
    domains: List[str] = None,
    ids: List[str] = None,
    limit: int = None,
    debug: bool = False,
):
    """
    Run evaluation on test data and save results in hierarchical format matching zero_shot_action.py.
    """
    # Load all JSON files from data directory
    files = sorted([f for f in glob.glob(os.path.join(data_dir, "*.json"))])
    if not files:
        print(f"No JSON files found in {data_dir}")
        return

    # Initialize structure
    all_results = {
        "summary": {},
        "domains": {}
    }
    
    # Save initialized empty structure immediately to overwrite old format
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Global accumulators for summary
    total_examples_all = 0
    total_correct_all = 0
    total_correct_with_plan_all = 0 # Will be 0
    total_plan_verified_all = 0 # Will be 0
    total_plan_yes_all = 0 # Count where GT is yes
    
    count = 0
    
    print(f"Starting evaluation on {len(files)} domain files...")
    
    for file_path in files:
        filename = os.path.basename(file_path).replace('.json', '')
        domain = filename.split('-')[0]
        if filename.startswith("blocksworld-4ops"):
             domain = "blocksworld-4ops"
        
        # Filter domains
        if domains and domain not in domains:
            continue
            
        print(f"\nProcessing domain: {domain} (File: {filename})")
        
        # Initialize domain stats
        if domain not in all_results["domains"]:
            all_results["domains"][domain] = {
                "total_examples": 0,
                "correct_answer": 0,
                "correct_with_plan": 0, # N/A but keeping structure
                "plan_verified": 0, # N/A
                "total_yes_examples": 0,
                "accuracy": 0.0,
                "plan_accuracy": 0.0,
                "pddl_domain": "N/A (No Domain Context Used)",
                "valid_actions": [],
                "details": []
            }
        
        domain_res = all_results["domains"][domain]

        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Iterate over examples
        for ex in data:
            example_id = ex.get("id", ex.get("example_id", f"example_{count}"))
            
            # Filter IDs
            if ids and str(example_id) not in ids:
                continue
                
            if limit and count >= limit:
                break
                
            count += 1
            print(f"Processing example {count} ({domain}/{example_id})...")
            
            context = ex.get("context", "")
            question = ex.get("question", ex.get("inputs", "")) # Handle question or inputs
            
            # Ensure ground truth is "yes" or "no"
            ground_truth = str(ex.get("answer", "")).lower()
            
            try:
                # Run DSPy module
                module = ZeroShotModule()
                response = module(
                    context=context,
                    question=question
                )
                
                predicted_answer = response.final_answer.strip().lower()
                reasoning = response.reasoning
                
                # In this simplified zero-shot, we don't produce plans
                initial_state = "N/A"
                goal_state = "N/A"
                actions = "N/A"
                # Verification is not applicable
                verification = {"valid": None, "reason": "No plan generated in simplified zero-shot"}
                
                if debug:
                    print(f"  Predicted: {predicted_answer}, Ground truth: {ground_truth}")
                
                # Determine correctness
                is_correct = predicted_answer == ground_truth
                is_yes_gt = ground_truth == "yes"
                
                # Plan related flags
                is_correct_with_plan = False # Never producing plan -> False
                plan_valid_display = "N/A"
                executed_final_state = "N/A"
                failure_reason = "No plan generated"
                
                # Update Domain Stats
                domain_res["total_examples"] += 1
                if is_correct:
                    domain_res["correct_answer"] += 1
                
                if is_yes_gt:
                    domain_res["total_yes_examples"] += 1
                
                # Recalculate accuracies
                if domain_res["total_examples"] > 0:
                    domain_res["accuracy"] = domain_res["correct_answer"] / domain_res["total_examples"]
                if domain_res["total_yes_examples"] > 0:
                    domain_res["plan_accuracy"] = 0.0 # No plans
                
                # Construct result item
                item = {
                    "id": example_id,
                    "context": context,
                    "question": question,
                    "gt_answer": ground_truth,
                    "pred_answer": predicted_answer,
                    "correct": is_correct,
                    "reasoning": reasoning,
                    "plan_valid": plan_valid_display,
                    "generated_plan": actions,
                    "generated_initial_state": initial_state,
                    "generated_goal_state": goal_state,
                    "executed_final_state": executed_final_state,
                    "plan_failure_reason": failure_reason,
                    "verification": verification
                }
                
                domain_res["details"].append(item)
                
                # Update Global Stats
                total_examples_all += 1
                if is_correct: total_correct_all += 1
                if is_yes_gt:
                    total_plan_yes_all += 1
                
                # Update Summary
                all_results["summary"] = {
                     "overall_accuracy": total_correct_all / total_examples_all if total_examples_all else 0,
                     "overall_accuracy_with_plan": 0.0,
                     "overall_plan_accuracy": 0.0,
                     "total_examples": total_examples_all,
                     "total_correct": total_correct_all,
                     "total_correct_with_plan": 0,
                     "total_plan_verified": 0,
                     "total_plan_possible": total_plan_yes_all
                }
                
                # Save incrementally
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                
            except Exception as e:
                print(f"  Error: {e}")
                pass
            
        if limit and count >= limit:
            break
    
    # Final print
    print(f"\nEvaluation Complete.")
    print(json.dumps(all_results["summary"], indent=2))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Zero-shot reasoning (Context + Question only)")
    parser.add_argument("--data", type=str, default="data/test_baseline", help="Path to test data directory")
    parser.add_argument("--output", type=str, default="baselines/zero_shot_results.json", help="Output path")
    parser.add_argument("--domains", type=str, nargs="+", help="Filter by domains")
    parser.add_argument("--id", type=str, nargs="+", help="Filter by example IDs")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--model-type", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--model-name", type=str, default="gpt-5.1")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Setup model
    setup_model(args.model_type, args.model_name)
    
    # Run evaluation
    run_evaluation(
        data_dir=args.data,
        output_path=args.output,
        domains=args.domains,
        ids=args.id,
        limit=args.limit,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
