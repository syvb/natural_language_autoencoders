#!/usr/bin/env bash
# Fetch the third-party A/B datasets used by v2 (not committed; ~17 MB).
# Sycophancy + neuroticism come from anthropics/evals (Perez et al. 2022, model-written evals).
# The yellow dataset is in-repo (likes_yellow_1000.json) and also at hf.co/datasets/syvb/yellow.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
base="https://raw.githubusercontent.com/anthropics/evals/main"
curl -fsSL "$base/sycophancy/sycophancy_on_nlp_survey.jsonl"            -o data/sycophancy_on_nlp_survey.jsonl
curl -fsSL "$base/sycophancy/sycophancy_on_political_typology_quiz.jsonl" -o data/sycophancy_on_political_typology_quiz.jsonl
curl -fsSL "$base/persona/neuroticism.jsonl"                            -o data/neuroticism.jsonl
echo "fetched: $(wc -l data/*.jsonl | tail -1)"
