#!/bin/bash
# Sequential, not parallel: three concurrent elicitations would contend on the
# same provider rate limit and on this machine's memory, and a stalled one would
# be invisible behind the others.
set -u
export LLM_CLIENT_PROJECT_MONTHLY_BUDGET=6.00
for M in openrouter/openai/gpt-5.6-luna openrouter/z-ai/glm-5.2 openrouter/deepseek/deepseek-v4-flash; do
  SHORT="${M##*/}"
  echo "=== $SHORT ==="
  ../.venv/bin/python scale_elicit.py --model "$M" --n-concepts 40 \
    --batch 45 --permutations 3 --out "results/scaled_${SHORT}.json" \
    2>&1 | grep -vE "^(Waiting|LLM_CLIENT|ROUTE|call_llm)"
done
