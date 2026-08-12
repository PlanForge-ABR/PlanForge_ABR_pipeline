#!/bin/bash

# Main execution script for NL2State processing
# This script runs nl2state.py, search.py, and results.py for all domains
# Then runs nl2state_2.py, search.py, and results.py for all domains

# Define all domains
# DOMAINS=("blocksworld" "ferry" "floortile" "grid" "grippers" "logistics" "rovers" "visitall" "alfworld" "depot" "satellite" "swap" "goldminer")
DOMAINS=("alfworld" "swap" "goldminer")

# Configuration
TRAIN_PATH="./data/train"
N=20
N_TRAIN=4
N_TRAIN_ABHILATION=1
N_VALIDATION=10
MODEL="openai"
MODEL_NAME="gpt-5.1"
TIMEOUT=60

echo "==============================================="
echo "Starting NL2State Pipeline - Phase 1"
echo "Using nl2state.py for domain-specific optimization"
echo "==============================================="
echo ""

# First loop: Execute nl2state.py, search.py, results.py for each domain
for domain in "${DOMAINS[@]}"; do
    echo "-----------------------------------------------"
    echo "Processing domain: $domain (Phase 1)"
    echo "-----------------------------------------------"
    
    # Step 1: Run nl2state.py
    echo "Step 1: Running nl2state.py for $domain..."
    python src/nl2state.py \
        --train "$TRAIN_PATH" \
        --N $N \
        --N_train $N_TRAIN \
        --N_validation $N_VALIDATION \
        --domain "$domain" \
        --out nl2state_result.json \
        --model "$MODEL" \
        --model_name "$MODEL_NAME"
    
    if [ $? -ne 0 ]; then
        echo "Error: nl2state.py failed for $domain"
        continue
    fi
    echo "✓ nl2state.py completed for $domain"
    echo ""
    
    # Step 2: Run search.py
    echo "Step 2: Running search.py for $domain..."
    python src/search.py \
        --domain "$domain" \
        --timeout $TIMEOUT \
        --src ./src \
        --out search_result.json
    
    if [ $? -ne 0 ]; then
        echo "Error: search.py failed for $domain"
        continue
    fi
    echo "✓ search.py completed for $domain"
    echo ""
    
    # Step 3: Run results.py after each domain
    echo "Step 3: Running results.py for $domain..."
    python src/results.py --split train
    
    if [ $? -ne 0 ]; then
        echo "Error: results.py failed for $domain"
        continue
    fi
    echo "✓ results.py completed for $domain"
    echo ""
    
    echo "✓✓✓ Phase 1 completed for $domain ✓✓✓"
    echo ""
done

echo "==============================================="
echo "Phase 1 Complete - All domains processed with nl2state.py"
echo "==============================================="
echo ""
echo ""

echo "==============================================="
echo "Starting NL2State Pipeline - Phase 2"
echo "Using nl2state_2.py for multi-domain optimization"
echo "==============================================="
echo ""

# # Second loop: Execute nl2state_2.py, search.py, results.py for each domain
# for domain in "${DOMAINS[@]}"; do
#     echo "-----------------------------------------------"
#     echo "Processing domain: $domain (Phase 2)"
#     echo "-----------------------------------------------"
    
#     # Step 1: Run nl2state_2.py
#     echo "Step 1: Running nl2state_2.py for $domain..."
#     python src/nl2state_2.py \
#         --train "$TRAIN_PATH" \
#         --N $N \
#         --N_validation $N_VALIDATION \
#         --domain "$domain" \
#         --out nl2state_result_abhilation.json \
#         --model "$MODEL" \
#         --model_name "$MODEL_NAME"\
    
#     if [ $? -ne 0 ]; then
#         echo "Error: nl2state_2.py failed for $domain"
#         continue
#     fi
#     echo "✓ nl2state_2.py completed for $domain"
#     echo ""
    
#     # Step 2: Run search.py
#     echo "Step 2: Running search.py for $domain..."
#     python src/search.py \
#         --domain "$domain" \
#         --timeout $TIMEOUT \
#         --src ./src \
#         --nl2state nl2state_result_abhilation.json \
#         --out search_result_abhilation.json
    
#     if [ $? -ne 0 ]; then
#         echo "Error: search.py failed for $domain"
#         continue
#     fi
#     echo "✓ search.py completed for $domain"
#     echo ""
    
#     # Step 3: Run results.py after each domain
#     echo "Step 3: Running results.py for $domain..."
#     python src/results.py \
#         --split train \
#         --out search_result_abhilation.json \
#         --summary_out accuracy_summary_abhilation.json \
#         --dom_summary_out search_accuracy_abhilation.json

#     if [ $? -ne 0 ]; then
#         echo "Error: results.py failed for $domain"
#         continue
#     fi
#     echo "✓ results.py completed for $domain"
#     echo ""
    
#     echo "✓✓✓ Phase 2 completed for $domain ✓✓✓"
#     echo ""
# done

echo "==============================================="
echo "Phase 2 Complete - All domains processed with nl2state_2.py"
echo "==============================================="
echo ""

echo "╔═══════════════════════════════════════════════╗"
echo "║   ALL PROCESSING COMPLETE!                    ║"
echo "║   Check accuracy_summary.json for results     ║"
echo "╚═══════════════════════════════════════════════╝"
