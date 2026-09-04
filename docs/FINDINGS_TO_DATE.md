# FINDINGS TO DATE

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
- Only **IsA** survives the region-matched null and FDR among relations, and it
  survives in *both* layer configurations (q=0.020 each), with its effect over
  the null rising from 0.230 to 0.251 when early layers are excluded — so it is
  not an embedding artefact. Within a system relations do behave consistently;
  across systems they largely do not.
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
