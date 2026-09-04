# FINDINGS TO DATE

## Run 002 happened on 2026-09-04 — scale ladder

Nine models, three families at three scales (0.5B / 1.6B / 3B), two languages,
eighteen systems. Preregistered in D19. Full write-up in `RUN_002_RESULTS.md`.

- **Nesting rises with scale.** Recovery toward perfect nesting goes 34.4% →
  39.2% → 39.7% at 20 clusters, monotone at all five resolutions tested, 15/15
  pairs beating the null in every cell, and holding under average, complete and
  ward linkage. This is the lattice account's own prediction and it survived.
- **Per-pair relation transfer stays at zero.** 0 of 10 relations significant at
  every tier. Best p improves 0.199 → 0.188 → 0.093 without crossing. Six-fold
  scale did not make a relation portable.
- **Group-level coherence replicates and stays flat**: 7 of 10 relations at every
  tier, mean effect 0.082 → 0.077 → 0.093.
- **Raw agreement drifts up**: mean held-out rho 0.369 → 0.371 → 0.398, 15/15
  pairs beating the null throughout; kNN stability 0.235 → 0.250 → 0.257.
- **A precision defect was found and corrected**: the large tier had silently run
  in float32 while the others ran in bfloat16, because `--dtype` applied only on
  CUDA and 3B does not fit this GPU. Re-extracted with precision genuinely held
  constant, the trend is unchanged (39.7% vs 39.4% at 20 clusters).
- **Net:** concepts converge with scale, relations do not. Neither reading of the
  run-001 fork wins outright.

## Run 001 happened on 2026-09-03

Superseding this document's previous headline, which said no real
pretrained-model result existed. One now does. Full write-up in
`RUN_001_RESULTS.md`, preserved outputs in `../results/run_001/`.

Verdict against `RESULT_INTERPRETATION_PROTOCOL.md`: **Tier 1** — systems share
some concept organization; reusable cross-system semantic transformations are
not established.

- Held-out cross-system RDM agreement beat an identity-permutation null in
  **14 of 15** system pairs (mean rho 0.273), and in **10 of 15** when layer
  selection was restricted away from the embedding and early layers
  (mean rho 0.190). The effect shrinks by about a third and does not vanish.
- Held-out neighbourhood stability: 0.2095 vs a 0.0661 null (z=61.7); 0.1402 vs
  0.0662 (z=32.3) under the restricted-layer re-analysis.
- **Unconstrained layer selection chose layers 0-3 for five of six systems**,
  and layer 0 — the embedding matrix — for XGLM/en. This was named as a risk in
  the README before the run and it happened, which is why the restricted re-run
  is the load-bearing number.
- Splitting by depth separates the pairs **by language, not by model**: all
  three Chinese cross-model pairs hold at depth (0.45, 0.40, 0.25, all q=0.003)
  while two of three English pairs lose significance. Unexpected, and a lead
  rather than a result.
- **WITHDRAWN 2026-09-04.** This bullet previously reported that only `IsA`
  survived its null and FDR. The statistic behind it is invariant to which source
  is paired with which target, so it could not test a relation at all; see D17.
  Corrected result, same extraction and seed: **no relation** lets a specific pair
  be carried to another model (`IsA`, `Causes` and `AtLocation` sit significantly
  *below* a distance-matched comparison), while **7 of 10 relations share a
  group-level direction** across three models and two languages at q=0.0014. The
  operator exists as an average, not as an instance.
  Within a system, relations still behave consistently: `Antonym`, `AtLocation`
  and `PartOf` clear the strictest permuted-pairing null in 6 of 6 systems, and
  that remains the strongest relation evidence in the run.
- The restricted-layer re-analysis selected the **final** layer for five of six
  systems, not a mid-depth one. These models agree most at their first and last
  layers and least in the middle.
- **The surface-confound control passes.** Regressing out character length,
  token count, token-id overlap and a frequency-rank proxy from both systems'
  tokenizers leaves **15 of 15** pairs significant at p=0.0005, costing about 5%
  of the effect (mean partial rho 0.2599 vs raw 0.2734) and 2% at the last
  layers. The convergence is not explained by shared spelling or tokenisation.
- **The static baseline passes too.** With fastText's aligned-vector geometry
  regressed out, LLM-LLM agreement falls only 0.2847 -> 0.2532 (76-99% retained
  per pair, 15/15 significant at p=0.0005). The convergence is not reducible to
  what a 2017 static embedding model already captures.
- **But the cross-language part is the weak part.** Same-language cross-model
  agreement is 0.3570; cross-language is 0.2365; aligned fastText manages 0.1861
  between the same two languages. The claim closest to "a language-independent
  semantic space" is the one least separable from a static baseline.
- The `--min-layer` sweep shows **no mid-depth layer is ever selected**: the
  selector jumps from the earliest layers straight to the last and stays there,
  flat at rho 0.187-0.200 for every cutoff from 2 to 24. The reported result does
  not depend on the arbitrary choice of 8.


## Added 2026-09-04

- **The headline had no denominator.** Extracting each system twice — bare word,
  then averaged over six templates — bounds how much agreement was reachable.
  `bloom_560m|en` scores 0.156 for agreeing with itself and is not a usable
  instrument; over the pairs where both systems are reliable, the models reach
  **0.468 of what was achievable** (0.615 within a language, 0.394 across).
- **Disagreement nests more than chance, weakly, and more so at finer
  resolution.** All 15 pairs beat a size-matched random-partition null; recovery
  toward perfect nesting runs 5.9% at 6 clusters (10/15 pairs) to 22.9% at 30
  (15/15). Predicted by `THEORY.md` §4 and reported in full at §7.
- **The three results above, the relation correction, and the map all say one
  thing:** these systems converge on local neighbourhoods and not on how
  neighbourhoods compose into a whole. Unanimous neighbour links are within a
  semantic region 76% of the time at mean map distance 0.20, against 43% and 0.42
  for links only two systems endorse.

## Instrument findings

Before run 001, synthetic tensors were used only to confirm that the code path
can:
- accept different hidden dimensions;
- select layers;
- construct RDM consensus;
- compute neighborhood statistics;
- compute relation signatures;
- run permutation/bootstrap/FDR procedures;
- generate reports.

Those values are not evidence about semantic space.

## Literature-level motivation

The cited literature supports investigating:
- partial multilingual internal alignment;
- representational convergence across some models/modalities;
- PRH as a hypothesis of increasing convergence;
- strong dataset/objective dependence as a caution;
- structured sperm-whale vocalizations as future adjacent context.

See `REFERENCES.md`.

## The null defect, found before run 001

This one is about the instrument, not about semantic space.

The relation test was a target-reuse detector. `random_target_pairs()` drew each
null pair's replacement target independently, which destroyed the target reuse
present in the observed probes: the original `IsA` set had 27 pairs over 9
distinct targets, seven of them ending at `mammal`, and those seven signatures
share the entire `d(mammal, .)` term while the null shared nothing. The null was
therefore strictly easier than the data for any relation that reused a target.

Fed **pure Gaussian random geometry**, containing no semantic content at all,
the test reported `IsA` at p <= .05 in **12 of 12 runs**, 11 of them surviving
Benjamini-Hochberg. The false-positive rate was monotone in the target-reuse
ratio and fell to nominal for relations whose targets were all distinct.

Verified on the original reuse-heavy probes, so the credit belongs to the null
fix and not to the benchmark rebuild:

| null | false positives / 12 runs on random geometry |
| --- | --- |
| original, independent resampling | 10 |
| multiplicity-preserving resampling | 0 |
| permuted pairing (both multisets exact) | 1 |

Two things are worth carrying forward. First, `KNOWN_RISKS_AND_OPEN_QUESTIONS.md`
items 8, 9 and 16 all circle this area — incomplete relation holdout, coarse
region matching, heterogeneous `IsA` — and none of them caught it. Reading the
design did not find the defect; running the pipeline on noise did, in about
twenty lines. Second, the defect was not in the enforcement machinery, which was
carefully built; it was entirely in what the null resampled.

## Engineering/design findings from this project

1. One global 2-D embedding is an inadequate definition of the target.
2. A multi-relational atlas is a better conceptual object.
3. Cross-model comparisons should use geometry/neighborhoods rather than raw neuron identity.
4. Coordinate-free relation signatures provide a plausible cross-model operator test.
5. Held-out layer selection is needed to reduce circularity.
6. Null models are required before interpreting convergence.
7. Hard matched nulls are more informative than only arbitrary random pairs.
7b. A null must preserve every structural property of the observed data that it
    is not testing. Preserving the multiplicity pattern of relation targets is
    load-bearing, not cosmetic.
7c. A statistical pipeline should be run on random inputs and required to report
    nothing, before it is run on real data.
8. Visualization should follow validated results.

These are design conclusions, not empirical discoveries about pretrained models.
