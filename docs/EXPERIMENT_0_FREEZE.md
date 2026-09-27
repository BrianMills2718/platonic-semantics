# EXPERIMENT 0 FREEZE RECORD

Created before the first real pretrained-model output, and re-frozen on
2026-09-03 twice: first after a defect was found in the relation null, then
again after the first real extraction exposed two more (bloom-560m returning NaN
in float16, and `learn`/`study` sharing one Chinese term). Both are recorded in
`FINDINGS_TO_DATE.md`. No real
run had occurred, so nothing confirmatory was invalidated: this record freezes
the corrected artifacts, and the hashes below supersede the earlier v2-handoff
record. The change rule at the bottom applies from here on.

The superseded artifacts, and why they changed, are in `FINDINGS_TO_DATE.md`
and `DECISION_LOG.md` (D16).

## Benchmark SHA-256

concepts.csv (212 concepts):
`73dbb915503fcb5e8c6e6c0bb5bba262c2d5785ee9e1a5a8699abcd74838b8db`

relations.csv (145 pairs, every relation type all-distinct targets):
`63510a698d04c1cd76e6d4c8f3cc347219de2c892d73913ebe73ae0b8cc570f1`

## Core code SHA-256

extract_representations.py:
`3d7b1bebfca2dbdb2d387fd0b1cb704946b68cadd8d742c21f429aa821845898`

analyze_semantic_geometry.py:
`9caab82176f223607d433c939954d6b527c88025c3a498329aac90396a5e92c5`

## Frozen pilot choices

Models:
- Qwen/Qwen2.5-0.5B
- bigscience/bloom-560m
- facebook/xglm-564M

Languages:
- en
- zh

Primary prompt:
- bare lexical term

Primary analysis:
- cosine RDM;
- stratified ~60/40 selection/evaluation concept split;
- kNN k=10;
- relation signature `S(a,b)[j] = d(b,j) - d(a,j)`;
- identity permutation geometry null;
- relabeled-system neighborhood null;
- random-target relation null (multiplicity-preserving);
- region-matched relation null (multiplicity-preserving);
- permuted-pairing relation null, the strictest within-system baseline;
- bootstrap uncertainty;
- Benjamini-Hochberg FDR.

## Change rule

If these benchmark or primary analysis choices change after seeing real headline outputs:
- preserve the original run;
- create a new experiment version;
- record why it changed;
- label the new analysis exploratory/corrected/replication as appropriate.

## Experiment 0.1 — corrected relation analysis, 2026-09-04

Created under the change rule above. Run 001's extraction, seed, split and
geometry results are unchanged and preserved in `results/run_001/primary/`; the
corrected relation analysis and the reliability ceiling are in
`results/run_001/reanalysis_20260904/` and are labelled **corrected**, not a new
run.

Why it changed: the cross-system relation statistic was algebraically invariant
to the source-target pairing and so could not test a transformation. Two
replacements were tried and rejected before the one that ships. D17 in
`DECISION_LOG.md` has the full account; `tests/test_pairing_sensitivity.py`
asserts each failure.

Benchmark: **unchanged.** Both hashes above still stand, and no probe, concept or
translation was touched.

### Core code SHA-256 (0.1)

extract_representations.py:
`3e1cc0eae755bcd556a888da1549166d65a49dcfe88dfde29830fb57db7878f1`

analyze_semantic_geometry.py:
`50e8c9bcb128331a27018f1d1f08ac3bb5166b1761ff7c14292765443cec0c4e`

### Primary analysis changes (0.1)

Added:
- pair-signature matrices with no self-exclusion, every pair scored on the same
  held-out anchor set;
- separation-matched random-pair null, holding cross-system mean source-target
  distance fixed;
- cross-system matched-pair agreement, tested against that null;
- cross-system relation coherence, tested against that null;
- prompt-formulation reliability ceiling.

Demoted to descriptive, retained in the outputs:
- cross-system mean-signature convergence (`relation_convergence.csv`).

Unchanged: extraction, seed 20260903, the 60/40 stratified split, cosine RDM,
kNN k=10, the identity-permutation geometry null, the relabeled-system
neighbourhood null, all three within-system relation nulls, bootstrap, and
Benjamini-Hochberg FDR.

## Experiment 0.2 — the scale ladder, 2026-09-04

Created under the change rule. Run 001's extraction and frozen outputs are
untouched in `results/run_001/`; run 002 is a new extraction at three scales and
is labelled a new experiment version, not a re-analysis.

Why it changed: D19. Run 001 left the project's central question forked, and
nothing in it separated "no shared global structure" from "0.5B is below the
resolution where it shows".

Benchmark: **unchanged.** Both hashes above still stand; no probe, concept or
translation was touched.

### Core code SHA-256 (0.2)

extract_representations.py:
`3e1cc0eae755bcd556a888da1549166d65a49dcfe88dfde29830fb57db7878f1`

analyze_semantic_geometry.py:
`545427d502ed5f307ff00a48180a97b89b5746a6bbf00a9f51a8bc21c63467f3`

### Model set (0.2)

Nine models, three families at three scales, two languages each.

| family | small | mid | large |
| --- | --- | --- | --- |
| Qwen2.5 | 0.5B | 1.5B | 3B |
| BLOOM | 560m | 1b7 | 3b |
| XGLM | 564M | 1.7B | 2.9B |

### Analysis changes (0.2)

- representations are mean-centred before the cosine RDM (`--center`, D19 primary);
  `--no-center` reproduces run 001 behaviour as a labelled sensitivity;
- `anisotropy()` is computed for every selected layer and written to
  `summary.json`, with any system above 0.95 listed in `degenerate_layer_flag`;
- `separation_matched_random_pairs` widens exhaustively over the full
  separation-ordered candidate list instead of a fixed 64-entry window. The
  truncated window emptied on the centred small tier and aborted the run;
  it raised rather than silently drawing an unmatched pair, which is why the
  defect was visible at all.

### Extraction changes (0.2)

- dtype is bfloat16 for all nine models so precision does not covary with scale.
  This required re-extracting the small tier, whose run-001 tensors were float32.
  bfloat16 keeps float32's exponent range, so the float16 NaN that hit BLOOM in
  run 001 cannot recur; all 18 tensors are verified finite.

Unchanged: seed 20260903, the 60/40 stratified split, k=10, 1000 permutations,
2000 bootstrap replicates, and every null.

## Experiment 0.2a — precision correction, 2026-09-04

Not a new experiment. Run 002's large tier is re-extracted because its compute
precision did not match its label, and the freeze is amended to the corrected
code. Benchmark unchanged.

**The defect.** `--dtype` was applied only when the device string began with
`cuda`, and documented that way. The large tier ran on CPU (3B does not fit a
4 GB GPU) and therefore loaded in float32 while the small and mid tiers ran in
bfloat16 — precision covarying with scale, which D19 fixed the dtype to prevent.
The per-model log line printed the requested dtype, not the loaded one, so
nothing surfaced it.

**Effect on the result: none.** Re-extracted with precision genuinely constant,
nesting at 20 clusters reads 34.7 / 39.2 / **39.7** against the original
34.7 / 39.2 / 39.4. Slightly higher, same direction, 15/15 pairs in every cell.

**Code changes.**
- `load_model` applies the requested dtype on every device, and raises if the
  loaded parameter dtype differs from the request.
- the saved log line reports the actual compute dtype.
- attention implementation is pinned to `eager` for every model and device. BLOOM
  supports nothing else, so it is the only setting the whole ladder can share; it
  also avoids an SDPA path that on this machine tries to build a triton CUDA
  helper and fails for want of Python.h.
- `scripts/nesting_by_scale.py` clips distances at zero before linkage, guarded by
  an assertion that the negative values are float noise rather than a real bug.

### Core code SHA-256 (0.2a)

extract_representations.py:
`ea1b7826f6af0556f5ff04782781548649c27269a3066eca50d8369b08ace832`

analyze_semantic_geometry.py:
`545427d502ed5f307ff00a48180a97b89b5746a6bbf00a9f51a8bc21c63467f3`

## Experiment 0.2b — templated-prompt pooling correction, 2026-09-27

Not a new experiment. Found by an external code audit.

**The defect.** With a templated `--prompt-mode` (e.g. `neutral`, "The concept
is {term}.") and no `--average-modes`, `main` sent the rendered prompts to
`encode_all`, which mean-pools every content token -- so the template's words
became part of every concept's representation. Term-span pooling
(`encode_term_in_context`) was used only on the averaging path, although its
own docstring states it is required whenever a template is added.

**Effect on the result: none.** Every saved representation was extracted with
`bare` (the template is the term itself, so pooling all content tokens is
correct and that path is unchanged) or with `--average-modes` (already
term-span). No `__neutral.npz` exists under `outputs/`.

**Code change.** A non-bare single template now uses `encode_term_in_context`.
`tests/test_audit_fixes.py` asserts both paths.

### Core code SHA-256 (0.2b)

extract_representations.py:
`12bfb1cb8f6768bc2bced41e349bd1cd60547aa06167376cfc3c328d87b6e0b9`

analyze_semantic_geometry.py:
`545427d502ed5f307ff00a48180a97b89b5746a6bbf00a9f51a8bc21c63467f3`
