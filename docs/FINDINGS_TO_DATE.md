# FINDINGS TO DATE

## There are currently NO real pretrained-model findings from this project

No Qwen/BLOOM/XGLM real-weight result is contained in this handoff.

Synthetic tensors were used only to confirm that the code path can:
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

## The one measured finding so far

It is about the instrument, not about semantic space, and it is the only
quantitative result this project has produced.

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
