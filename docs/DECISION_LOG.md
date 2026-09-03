# DECISION LOG / CONVERSATION HISTORY

This document records the major conceptual turns that led to the current design so a new coding agent does not reopen settled questions without understanding why they were settled.

## D1 — Treat Platonic semantic space as a hypothesis

The project began from reports of multilingual internal alignment, cross-model representational convergence, and adjacent discussion of sperm-whale communication.

Decision: test **invariant relational geometry**, not assume a universal latent language already exists.

## D2 — Whale communication is future scope

Sperm-whale vocalizations show contextual/combinatorial structure, but there is no demonstrated point-to-point semantic alignment with human language or LLM latent spaces.

Decision: whales enter only later through shared behavioral/context anchors.

## D3 — The map must be empirical

An early response produced a decorative illustration. The user clarified that the goal was a map generated from actual representations.

Decision: future scientific maps must derive from measured activations/distances unless visibly labeled demo/mockup.

## D4 — PRH/WIT prototype is a precursor, not the main experiment

An early runnable map used public PRH WIT features and consensus RDM geometry.

Decision: keep it as methodological background. The current experiment focuses on explicit semantic concepts and semantic relations.

## D5 — UI artifacts should be executable

Several image mockups were produced when the desired deliverable was an interactive artifact.

Decision: UI prototypes should be standalone HTML or equivalent, and synthetic data must be explicitly labeled.

## D6 — The mathematics project is separate

The user shared a separate project about mapping mathematical objects. It was useful food for thought but was mistakenly treated as the new subject.

Decision: this project remains about **general semantic space**.

Imported methodological lessons only:
- multiple relation types;
- multiple metrics/lenses;
- disagreement as information;
- atlas/local-chart metaphor;
- avoid one supposedly perfect 2-D map.

## D7 — Move from a concept cloud to a relational atlas

A generic embedding scatterplot still returned the project to its starting point.

Decision: focus on:
- stable neighborhoods;
- stable semantic transformations/operators;
- model/language/layer variation;
- uncertainty and disagreement.

## D8 — Raw displacement vectors are not cross-model objects

`x_b - x_a` cannot be directly compared between unrelated neural coordinate systems.

Decision: use coordinate-free relation signatures:

    S(a,b)[j] = d(b,j) - d(a,j)

where `j` indexes shared semantic anchors.

## D9 — External semantic graphs are probes, not ground truth

ConceptNet-style relation labels are useful for a first benchmark.

Decision: external graphs can label, probe, and validate; they do not define neural coordinates.

## D10 — Validation before further visualization

Once the relational experiment existed, statistical hardening became the priority.

Added:
- held-out concept split;
- identity permutation null;
- neighborhood null;
- random-target relation null;
- region-matched relation null;
- bootstrap intervals;
- FDR correction.

## D11 — Layer selection must be held out

Layers are chosen using only selection concepts.

Headline geometry evaluation uses separate evaluation concepts.

This is intended to reduce circularity.

## D12 — Current relation holdout is only partial

A strict relation-endpoint holdout left too few pairs.

Current pilot compromise:
- relation pairs may use the full probe list;
- relation signatures are compared on held-out anchor dimensions.

This is not equivalent to fully held-out relation examples.

## D13 — Hard matched nulls matter

Random targets can be too easy.

Decision: include broad-region-matched target nulls now; add frequency/POS/token-length matching later.

## D14 — Synthetic smoke tests are not findings

The pipeline was tested on synthetic tensors of different hidden dimensions.

The original environment could not run real pretrained weights.

Decision: never present those smoke-test values as semantic evidence.

## D16 — Relation nulls preserve target multiplicity; probes use distinct targets

D13 added region-matched target nulls on the reasoning that random targets can be
too easy. That was right about the direction and missed the mechanism. Every null
here resampled each pair's target *independently*, which destroys the target
reuse the observed probes have, and makes the null easier than the data exactly
where a relation reuses a target. On random geometry the test then reported `IsA`
in 12 of 12 runs (see `FINDINGS_TO_DATE.md`).

Decision, 2026-09-03:

1. Every relation null draws one replacement per **distinct** observed target and
   reuses it across the pairs that shared it, preserving the multiplicity pattern.
2. A permuted-pairing null is added for within-system consistency, keeping both
   observed multisets exactly. It is deliberately **not** used for cross-system
   mean-signature convergence, where the mean of `d(b,.) - d(a,.)` is very nearly
   invariant under a pairing permutation and the test would have no power by
   construction.
3. The probe set is rebuilt so every relation type has all-distinct targets. This
   is forced, not aesthetic: with reused targets the *correct* null converges on
   the observed data, so those relations are untestable rather than merely noisy.
   The concept inventory grew 174 -> 212 because the original basic-level nouns
   did not contain enough superordinates to build distinct-target `IsA` probes.
4. `target_reuse_ratio` is reported in every relation output, and
   `tests/test_null_calibration.py` runs the analysis on random geometry and
   fails if it reports a result.

This supersedes nothing in D10-D13 except the null's implementation; the
held-out split, permutation logic, bootstrap and FDR are unchanged.

## D15 — The first real pretrained-model run is the next milestone

The next milestone is:
1. real extraction;
2. QC;
3. quick statistical run;
4. serious statistical run;
5. conservative interpretation;
6. robustness;
7. atlas only if warranted.
