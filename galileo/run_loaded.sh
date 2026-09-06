#!/bin/bash
set -u
export LLM_CLIENT_PROJECT_MONTHLY_BUDGET=6.00
for M in openrouter/openai/gpt-5.6-luna openrouter/deepseek/deepseek-v4-flash openrouter/z-ai/glm-5.2; do
  S="${M##*/}"; echo "=== $S ==="
  ../.venv/bin/python scale_elicit.py --model "$M" --concepts concepts_loaded.csv \
    --n-concepts 40 --batch 45 --permutations 3 --out "results/loaded_${S}.json" 2>&1 \
    | grep -vE "^(Waiting|LLM_CLIENT|ROUTE|call_llm)"
done
