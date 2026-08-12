#!/usr/bin/env bash
# Run evaluate.py across all domains discovered in data/test.
# Configuration is driven by env vars: MODULE_SOURCE, MODULE_FILE, TEST_ROOT, TIMEOUT,
# STRATEGY, MODEL, MODEL_NAME, SRC_DIR, PYTHON_BIN.

set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE_SOURCE="${MODULE_SOURCE:-domain}"   # domain | multi
MODULE_FILE="${MODULE_FILE:-}"             # optional override for the optimised module filename
TEST_ROOT="${TEST_ROOT:-data/test_baseline}"
TIMEOUT="${TIMEOUT:-300}"
STRATEGY="${STRATEGY:-greedy_bfs}"
MODEL="${MODEL:-openai}"
MODEL_NAME="${MODEL_NAME:-gpt-5.1}"
MODEL_NAME="${MODEL_NAME:-grok-4-1-fast-non-reasoning}"
SRC_DIR="${SRC_DIR:-./src}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-}"
DOMAIN_OUTPUT_FILE="${DOMAIN_OUTPUT_FILE:-}"
OVERALL_OUTPUT_FILE="${OVERALL_OUTPUT_FILE:-}"

[[ -d "$TEST_ROOT" ]] || { echo "Error: test root not found: $TEST_ROOT" >&2; exit 1; }

if [ "$#" -gt 0 ]; then
    echo "Using specified domains: $*"
    DOMAINS=("$@")
else
    echo "Discovering domains under $TEST_ROOT..."
    DOMAIN_OUTPUT="$("$PYTHON_BIN" - "$TEST_ROOT" <<'PY'
import glob
import os
import sys

root = sys.argv[1]
found = set()
for p in glob.glob(os.path.join(root, "*.json")):
    name = os.path.basename(p)
    if name.endswith("-test.json"):
        found.add(name[:-10])
    elif "-test-dev" in name:
        found.add(name.split("-test-dev", 1)[0])
domains = sorted(found)
if not domains:
	sys.exit(f"No domains found under {root}")
print("\n".join(domains))
PY
    )" || { echo "Error: failed to discover domains" >&2; exit 1; }

    DOMAINS=()
    while IFS= read -r dom; do
        [[ -z "$dom" ]] && continue
        DOMAINS+=("$dom")
    done <<<"$DOMAIN_OUTPUT"
fi

if [ "${#DOMAINS[@]}" -eq 0 ]; then
	echo "Error: no domains found." >&2
	exit 1
fi

echo "Running evaluate.py for: ${DOMAINS[*]}"
overall_status=0

for domain in "${DOMAINS[@]}"; do
	echo "==> Domain: $domain (module_source=$MODULE_SOURCE)"

	cmd=(
		"$PYTHON_BIN" evaluate.py
		--domain "$domain"
		--module_source "$MODULE_SOURCE"
		--test_root "$TEST_ROOT"
		--timeout "$TIMEOUT"
		--strategy "$STRATEGY"
		--model "$MODEL"
		--model_name "$MODEL_NAME"
		--src "$SRC_DIR"
	)
	if [[ -n "$MODULE_FILE" ]]; then
		cmd+=(--module_file "$MODULE_FILE")
	fi

	if [[ -n "$OUTPUT_FOLDER" ]]; then
		cmd+=(--output_dir "$OUTPUT_FOLDER")
	fi
	if [[ -n "$DOMAIN_OUTPUT_FILE" ]]; then
		cmd+=(--domain_output_file "$DOMAIN_OUTPUT_FILE")
	fi
	if [[ -n "$OVERALL_OUTPUT_FILE" ]]; then
		cmd+=(--overall_output_file "$OVERALL_OUTPUT_FILE")
	fi

	if ! "${cmd[@]}"; then
		echo "  ERROR: evaluate.py failed for $domain" >&2
		overall_status=1
	fi
	echo
done

if [ $overall_status -eq 0 ]; then
	echo "All domains completed. Check src/accuracy_summary_test_${MODULE_SOURCE}.json for the aggregate results."
else
	echo "One or more domains failed. See logs above for details."
fi

exit $overall_status
