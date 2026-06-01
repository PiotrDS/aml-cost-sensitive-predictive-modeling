#!/usr/bin/env bash
set -euo pipefail

echo "Enter student IDs:"
read -rp "ID1: " ID1
read -rp "ID2: " ID2
read -rp "ID3: " ID3

STUDENT_IDS="${ID1}_${ID2}_${ID3}"

echo
echo "Running all models for student_ids=${STUDENT_IDS}"
echo

MODELS=("xgb" "lasso" "forward" "combined")

for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running model: ${MODEL}"
    echo "=========================================="

    uv run python code/main.py \
        --model "${MODEL}" \
        --student_ids "${STUDENT_IDS}"

    echo
done

echo "All models finished."
echo "Results should be saved in:"
echo "  submission/xgboost/"
echo "  submission/lasso/"
echo "  submission/forward/"
echo "  submission/combined/"