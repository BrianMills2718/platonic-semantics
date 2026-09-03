# Run 001 — first real pretrained-model results

**Date:** 2026-09-03
**Status:** complete; primary run plus one labelled exploratory sensitivity run
**Verdict against `RESULT_INTERPRETATION_PROTOCOL.md`: Tier 1.**

> Systems share some concept organization, but reusable cross-system semantic
> transformations are not established.

This is the first time any pretrained model has been run in this project.
Everything before it was design.

## What ran

Three model families, two languages, 212 concepts, 145 relation probes, on a
local NVIDIA T600 (4 GB), float32, ~2 minutes of extraction. 1000 permutations,
2000 bootstrap replicates.

The handoff said this needed Colab. It did not: that constraint belonged to the
environment the design was written in, not to the work. A 0.5B model at 212
short prompts is a laptop-GPU job.

## Two defects found before any result was believed

Neither was visible in the code. Both were found by QC on the extracted tensors,
which is now `scripts/qc_representations.py`.

1. **bloom-560m in float16 on CUDA returned NaN** — 80% of the English tensor.
   The extractor saved it without complaint, and `corr()` maps a non-finite
   result to `0.0`, which reads downstream as "these systems do not agree". A
   silent zero, not an error. float32 on the same GPU and inputs is clean
   (max|h| = 1380). The extractor now refuses to save non-finite output, and
   float32 is the default.
2. **`learn` and `study` both carried the Chinese term 学习**, so in every
   Chinese system their representations were byte-identical — a zero distance
   that says nothing about meaning, inside the matrix the whole comparison rests
   on. `study` is now 研究, and a test forbids shared surface forms.

A third, bounded and not fixed: **six Chinese terms (哺乳动物, 鲸, 锤子, 钥匙 and
two others) tokenise to `<unk>` in XGLM only**, 2.8% of the benchmark, making
them mutually indistinguishable in that one system. 鲸 (whale) and 烹饪 (cook)
sit at cosine 1.000000. Qwen and BLOOM cover the set fully.

## The headline number, and why it needs its caveat

Held-out cross-system RDM agreement, against an identity-permutation null:

| | pairs beating null (FDR q<=.05) | mean held-out Spearman rho |
|---|---|---|
| layer chosen freely | **14 / 15** | 0.273 |
| layer restricted to >= 8 | **10 / 15** | 0.190 |

Held-out neighbourhood stability, against a relabelled-system null:

| | observed | null | z | p |
|---|---|---|---|---|
| layer chosen freely | 0.2095 | 0.0661 | 61.7 | 0.001 |
| layer restricted to >= 8 | 0.1402 | 0.0662 | 32.3 | 0.001 |

**The caveat is the layer choice.** Unconstrained selection put five of six
systems at layers 0-3 of 25, and XGLM/en at **layer 0 — the embedding matrix
itself**. This was written into the README as a named risk before the run: *"if
they cluster at 0-2, the result is about tokenisers."* It happened.

So the sensitivity run above is the load-bearing one. Excluding the embedding
and every early layer, the effect shrinks by roughly a third and **does not
vanish**: 10 of 15 pairs still beat the null, and neighbourhood stability still
clears it at z=32. The convergence is not purely lexical.

## The finding that was not expected

Restricting to depth splits the pairs cleanly by language, not by model:

| pair | free | deep (>=8) | deep q |
|---|---|---|---|
| bloom·zh vs qwen·zh | 0.432 | **0.448** | 0.003 |
| qwen·zh vs xglm·zh | 0.426 | **0.397** | 0.003 |
| bloom·zh vs xglm·zh | 0.342 | **0.248** | 0.003 |
| qwen·en vs xglm·en | 0.345 | 0.311 | 0.003 |
| bloom·en vs qwen·en | 0.358 | 0.110 | 0.060 (n.s.) |
| bloom·en vs xglm·en | 0.237 | 0.088 | 0.089 (n.s.) |

**Chinese representations agree across models at depth. English ones largely do
not.** Two of three English cross-model pairs lose significance once early
layers are excluded, while all three Chinese pairs hold and the strongest one
gets stronger.

That is the opposite of the naive expectation — these models all saw far more
English. It is a single pilot at 0.5B on 212 bare words and should be treated as
a lead, not a result. Candidate explanations worth testing: English single words
are more polysemous and each model resolves them differently; Chinese words in
this set tokenise more consistently across these three tokenizers; or the
Chinese subset is small enough in training that all three models learned it from
similar sources.

Also worth noting from `cross_language_layer_curve.csv`: BLOOM and Qwen peak in
English/Chinese agreement at **layer 1** (0.252, 0.389), collapse through the
middle, and recover at the last layers. XGLM peaks at **layer 16** (0.273) —
the mid-depth profile you would want to see. The three models do not agree about
where cross-language structure lives.

## Relations

Only one relation survives the region-matched null and FDR correction in the
primary run:

| relation | cross-system similarity | effect over region-matched null | q |
|---|---|---|---|
| **IsA** | 0.561 | 0.230 | **0.020** |
| AtLocation | 0.466 | 0.058 | 0.095 |
| HasProperty | 0.770 | 0.016 | 0.113 |
| PartOf | 0.468 | 0.037 | 0.684 |
| everything else | — | <= 0 | ~1.0 |

Every relation now has `target_reuse_ratio` 1.00, so this is not the artefact
that the earlier null produced. Note `HasProperty`: the highest raw similarity
in the table (0.770) and almost no effect over its null. Raw similarity is not
evidence, which is the entire reason the nulls exist.

Within a system, relations do behave like consistent transformations —
`Antonym`, `AtLocation` and `PartOf` clear the strictest permuted-pairing null
in 6 of 6 systems. What does not replicate is the same transformation appearing
in *different* systems.

## What this does not show

- Not a Platonic semantic space, not a language-independent internal language.
- No non-neural baseline was run. The identity-permutation null only asks
  whether *any* shared structure exists; a static-embedding or co-occurrence
  floor would ask the question that matters. Until that exists these numbers
  cannot be compared against the trivial explanation.
- 0.5B models, bare single words, one prompt template, one split seed, one k.
- 2.8% of the Chinese benchmark is `<unk>` in XGLM.
- The mid-depth cutoff of 8 was chosen after seeing the primary layers. It is a
  labelled exploratory sensitivity analysis, not a preregistered choice.

## Next, in order

1. A fastText EN/ZH baseline and a lexical-confound RDM (token length,
   frequency). Without it the headline is not falsifiable.
2. Re-run at several `--min-layer` values rather than one, and report the curve.
3. Prompt-averaged representations instead of bare words.
4. Fix or exclude the six XGLM `<unk>` concepts.
