#!/bin/bash
cd /home/brian/code/platonic-semantics
SMALL="qwen25_05b xglm_564m"
MID="qwen25_15b bloom_1b7 xglm_17b"
LARGE="qwen25_3b bloom_3b xglm_29b"
for tier in small mid large; do
  case $tier in small) MS="$SMALL";; mid) MS="$MID";; large) MS="$LARGE";; esac
  for m in $MS; do
    for mode in bare averaged; do
      out=outputs/run003/$mode
      if ls $out/${m}__zh__*.npz >/dev/null 2>&1; then echo "skip $m $mode"; continue; fi
      echo "=== $tier $m $mode $(date +%H:%M:%S) ==="
      if [ "$mode" = bare ]; then
        .venv/bin/python src/extract_representations.py --models $m --prompt-mode bare \
          --dtype bfloat16 --device cpu --outdir $out 2>&1 | grep -E "saved|Error|RuntimeError" | tail -2
      else
        .venv/bin/python src/extract_representations.py --models $m --average-modes config \
          --dtype bfloat16 --device cpu --outdir $out 2>&1 | grep -E "saved|Error|RuntimeError" | tail -2
      fi
    done
  done
  echo "TIER_${tier}_DONE $(date +%H:%M:%S)"
done
echo RUN003_DONE
