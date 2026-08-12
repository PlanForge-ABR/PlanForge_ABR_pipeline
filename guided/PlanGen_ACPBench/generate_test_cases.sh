#!/usr/bin/env bash
# Run all unit test generators for all domains and save into src/<domain>/.
# Fixed per request:
#   NL2State: N=20, seed=1
#   Succ:     N=20, seed=1
#   Goal:     N=20, seed=1
# Train dir defaults to ./data/train but can be overridden by $TRAIN_DIR

set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DIR="${TRAIN_DIR:-./data/train}"

GEN_DIR="test_case_generator"
GEN_NL2="${GEN_DIR}/unit_test_generator_nl2state.py"
GEN_SUCC="${GEN_DIR}/unit_test_generator_succ.py"
GEN_GOAL="${GEN_DIR}/unit_test_generator_goal.py"

N_NL2=20
N_SUCC=20
N_GOAL=20
SEED=1

# DOMAINS=(
#   blocksworld
#   ferry
#   floortile
#   grid
#   grippers
#   logistics
#   rovers
#   visitall
#   depot
#   satellite
#   swap
#   goldminer
#   frogs_jumping
#   hanoi
# )
DOMAINS=(
  frogs_jumping
)

# Checks
for f in "$GEN_NL2" "$GEN_SUCC" "$GEN_GOAL"; do
  [[ -f "$f" ]] || { echo "Error: generator not found: $f" >&2; exit 1; }
done
[[ -d "$TRAIN_DIR" ]] || { echo "Error: training dir not found: $TRAIN_DIR" >&2; exit 1; }

echo "Using:"
echo "  TRAIN_DIR = $TRAIN_DIR"
echo "  SEED      = $SEED"
echo "  N_NL2     = $N_NL2"
echo "  N_SUCC    = $N_SUCC"
echo "  N_GOAL    = $N_GOAL"
echo

for d in "${DOMAINS[@]}"; do
  OUT_DIR="src/${d}"
  mkdir -p "$OUT_DIR"

  echo "==> Domain: $d"

  # NL2State: N_NL2 tests
  echo "  [NL2State] -> ${OUT_DIR}/nl2state_test.json"
  if ! "$PYTHON_BIN" "$GEN_NL2" \
    --train "$TRAIN_DIR" \
    --N "$N_NL2" \
    --domain "$d" \
    --out "${OUT_DIR}/nl2state_tests.json" \
    --seed "$SEED"; then
    echo "    ERROR: NL2State generator failed for domain $d" >&2
    continue
  fi

  # Succ: N_SUCC tests
  echo "  [Succ]     -> ${OUT_DIR}/succ_tests.json"
  if ! "$PYTHON_BIN" "$GEN_SUCC" \
    --train "$TRAIN_DIR" \
    --N "$N_SUCC" \
    --domain "$d" \
    --out "${OUT_DIR}/succ_tests.json" \
    --seed "$SEED"; then
    echo "    ERROR: Succ generator failed for domain $d" >&2
    continue
  fi

  # Goal: N_GOAL tests
  echo "  [Goal]     -> ${OUT_DIR}/goal_tests.json"
  if ! "$PYTHON_BIN" "$GEN_GOAL" \
    --train "$TRAIN_DIR" \
    --N "$N_GOAL" \
    --domain "$d" \
    --out "${OUT_DIR}/goal_tests.json" \
    --seed "$SEED"; then
    echo "    ERROR: Goal generator failed for domain $d" >&2
    continue
  fi

  echo
done

echo "All done ✅"
