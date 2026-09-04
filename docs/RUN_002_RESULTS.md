# Run 002 — does convergence rise with scale?

**Date:** 2026-09-04
**Design:** preregistered in D19 before any tensor was extracted
**Verdict: the preregistered prediction is half met, and neither reading wins.**

Run 001 left the project's central question forked (`THEORY.md` §7): either there
is no shared global structure at this level of description, or 0.5B models on
bare single words sit below the resolution where it appears. This run tests it.

## What ran

Nine models, three families at three scales, two languages, 212 concepts, 145
relation probes. Eighteen systems, 15 cross-system pairs per tier.

| family | small | mid | large |
| --- | --- | --- | --- |
| Qwen2.5 | 0.5B | 1.5B | 3B |
| BLOOM | 560m | 1b7 | 3b |
| XGLM | 564M | 1.7B | 2.9B |

A six-fold scale range. Within-family comparison controls architecture and
training data; cross-family comparison at each tier is the convergence
measurement. Everything else is run 001's design unchanged: seed 20260903, the
60/40 stratified split, k=10, 1000 permutations, 2000 bootstrap replicates, every
null.

Two things were fixed in advance because they would otherwise confound the whole
comparison. Representations are mean-centred before the cosine RDM, decided on
principle in D19 rather than after seeing results. Dtype is bfloat16 for all nine
models so precision does not covary with scale, which required re-extracting the
small tier. All 18 tensors verified finite; the only duplicate rows are XGLM's
known `whale`/`cook` `<unk>` collision, identical at every XGLM scale because the
tokenizer does not change with model size.

## The preregistered prediction, and what happened

D19 named two primary outcomes and one secondary, and said which pattern would
support which reading. Reproduced here as written, before the numbers existed:

> **If reading (b) is right** — models are coarsenings and 0.5B is below the
> resolution — then with scale: nesting recovery rises monotonically, and
> per-pair relation transfer becomes non-zero for at least one relation.
>
> **If reading (a) is right**, cross-family agreement may drift up while nesting
> stays flat and per-pair transfer stays at zero.

### Primary 1 — nesting. Rises. Prediction met.

Recovery toward perfect nesting, against a size-matched random-partition null:

| clusters | small (~0.5B) | mid (~1.6B) | large (~3B) |
| --- | --- | --- | --- |
| 6 | 25.6% | 26.0% | **30.4%** |
| 10 | 29.7% | 32.5% | **36.3%** |
| 14 | 33.3% | 37.3% | **37.5%** |
| 20 | 34.7% | 39.2% | **39.4%** |
| 30 | 34.3% | 38.2% | **39.7%** |

Monotone at every resolution, 15 of 15 pairs beating the null in all fifteen
cells. Reproduce with `python scripts/nesting_by_scale.py`; the numbers above are
that script's output, and `results/run_002/nesting_by_scale.json` is the record.

It is not an artefact of one clustering method — at k=20 the trend holds under
average (34.7 → 39.2 → 39.4), complete (25.0 → 27.3 → 27.1) and ward
(30.4 → 34.4 → 35.6) linkage. Absolute values differ by method; the direction
does not. The complete-linkage tier ordering is the one wobble: mid and large are
within 0.2 points of each other, so that method shows small → {mid, large} rather
than a clean three-step rise.

### Primary 2 — per-pair relation transfer. Still zero. Prediction not met.

Can a specific relation pair be carried to another model, beyond what two
arbitrary concepts at the same separation give you?

| tier | relations significant (q ≤ .05) | best p | best relation | mean effect |
| --- | --- | --- | --- | --- |
| small | **0 / 10** | 0.199 | Associated | −0.026 |
| mid | **0 / 10** | 0.188 | HasProperty | −0.026 |
| large | **0 / 10** | 0.093 | HasProperty | −0.031 |

Nothing reaches significance at any scale. The best p-value improves in a
consistent direction (0.199 → 0.188 → 0.093) but does not cross, and the mean
effect stays negative throughout. Across a six-fold scale range, no relation
became a portable per-instance operation.

### Secondary — raw agreement. Rises modestly.

| tier | cross-family same-language | cross-language | all 15 pairs | held-out kNN stability |
| --- | --- | --- | --- | --- |
| small | 0.478 | 0.285 | 0.369 | 0.235 vs 0.066 null |
| mid | 0.485 | 0.285 | 0.371 | 0.250 vs 0.066 |
| large | 0.517 | 0.311 | 0.398 | 0.257 vs 0.066 |

15 of 15 pairs beat the identity-permutation null at every tier. D19 called this
the weak discriminator in advance, and it behaves accordingly: it drifts up under
either reading and settles nothing.

### Coherence — flat.

Group-level shared direction holds at **7 of 10 relations at every tier**, with
mean effect 0.082 → 0.077 → 0.093. The group-average result from run 001
replicates at all three scales and does not strengthen with scale.

## What this means

**Neither reading wins, and the split is informative.** The structural measure
behaves as (b) predicts — models look more like coarsenings of one structure as
they get larger. The relational measure behaves as (a) predicts — the operator
story stays dead across the whole range. Reporting this as confirmation of (b)
would require ignoring the outcome D19 named as primary alongside it.

The most defensible reading of all four measurements together: **concepts
converge with scale; relations do not.** Larger models agree more about which
things belong together, and agree in a more nested way, while remaining unable to
transport a specific relation instance between them. Whatever a relation is in
these representations, it is a property of a set and not of a pair, and six-fold
scale does not change that.

## What would settle it

The per-pair trend is monotone in p but far from significance, so the honest
statement is that this range cannot decide it. Options in order of expected
value:

1. **Extend the ladder upward.** 7B is the next rung all three families publish.
   If per-pair transfer crosses there while nesting keeps climbing, (b) is
   confirmed. If nesting keeps climbing and transfer stays flat, "concepts
   converge, relations do not" becomes the finding rather than the interim
   reading.
2. **Contextualised stimuli.** Every result here rests on bare single words. A
   relation may only be a portable operation when its arguments are disambiguated
   by context, which this design cannot see.
3. **A training-seed ceiling.** Still owed from run 001 and still the better
   reliability bound than the prompt-formulation proxy.

## Limits specific to this run

- Three points on a six-fold range is a short ladder; monotone across three
  points is suggestive, not a curve.
- Model families are not independent of scale: each family's larger members share
  a tokenizer and much training data with its smaller ones, so within-family
  comparisons are not clean replicates.
- One system per tier still trips the anisotropy flag — `xglm_564m|zh` (0.975),
  `xglm_17b|zh`, and both `xglm_29b` systems. Centering moved layer selection
  away from the near-degenerate deep layers for the other families, which is why
  five flags in run 001 became one or two here.
- The prompt-reliability floor from run 001 was not re-measured for run 002's
  systems; no prompt-averaged extraction exists at the mid and large tiers.
- Everything inherits run 001's standing limits: bare single words, one split
  seed, 14–18 pairs per relation, 15 non-independent system pairs.
