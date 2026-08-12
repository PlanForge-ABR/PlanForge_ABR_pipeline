#!/usr/bin/env python3
"""
Regenerate search accuracy files for all domains using the correct test data root.
This fixes issues where ground truth was missing (null) in the results.
"""

import os
import sys
from src import results

def main():
    # Configuration matching evaluate.py
    test_root = "data/test_baseline"
    module_source = "domain" # or 'multi' - strictly evaluate.sh uses 'domain' by default
    
    # Filenames from evaluate.py
    mode_tag = module_source # strictly evaluate.py logic: "domain" if args.module_source == "domain" else "multi"
    search_result_filename = f"search_result_test_{mode_tag}.json"
    summary_out = f"accuracy_summary_test_{mode_tag}.json"
    dom_summary_out = f"search_accuracy_test_{mode_tag}.json"

    print(f"Fixing results in src/ using ground truth from {test_root}...")
    print(f"Targeting result files: {search_result_filename}")

    if not os.path.exists(test_root):
        print(f"Error: Test root {test_root} does not exist.")
        sys.exit(1)

    results.main(
        split="test",
        omit_per_example=False,
        out=search_result_filename,
        summary_out=summary_out,
        dom_summary_out=dom_summary_out,
        data_root=test_root
    )

    print("Done. Check src/accuracy_summary_test_domain.json")

if __name__ == "__main__":
    main()
