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

uv run python code/main.py \
    --model all \
    --student_ids "${STUDENT_IDS}"

echo
echo "All models finished."
echo "Leaderboard files should be saved in submission/leaderboard/."
