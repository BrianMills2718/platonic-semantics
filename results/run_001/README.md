# Run 001 — preserved outputs

The analysis writes to `outputs/`, which is gitignored and overwritten by the
next run. These are the committed copies of the first real pretrained-model run,
so the numbers in `docs/RUN_001_RESULTS.md` can be checked against the files
that produced them.

- `primary/` — layer selection unconstrained. The preregistered configuration.
- `sensitivity_minlayer8/` — layer selection restricted to layer >= 8. A
  labelled **exploratory** re-analysis, chosen after seeing that the primary run
  put five of six systems at layers 0-3. Per
  `docs/RESULT_INTERPRETATION_PROTOCOL.md` it does not replace the primary
  result.

Extraction: 2026-09-03, local NVIDIA T600, float32, transformers 5.16.1,
torch 2.6.0+cu124. Benchmark and code hashes are pinned in
`docs/EXPERIMENT_0_FREEZE.md`.

Representations themselves (~50 MB of .npz) are not committed; regenerate with
`python src/extract_representations.py --device cuda`, then verify with
`python scripts/qc_representations.py`.
