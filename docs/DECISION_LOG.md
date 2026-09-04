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

## D17 — The cross-system relation test is replaced; D16.2 drew the wrong conclusion

D16 item 2 noticed that the cross-system mean-signature statistic is "very nearly
invariant under a pairing permutation" and concluded that the permuted-pairing
null should be skipped there for lack of power. That reasoning inverts the
finding. The invariance is not a property of the null. It is a property of the
statistic, and it is exact rather than approximate:

```text
mean_i( d[b_i,.] - d[a_i,.] )  ==  mean(d[targets,.]) - mean(d[sources,.])
```

The pairing cancels algebraically. Verified 2026-09-04 by removing the
`sig[a] = sig[b] = nan` self-exclusion and permuting the pairing: twenty
permutations reproduce run 001's observed `IsA` value to eight decimal places
(0.6835608). The residual sensitivity visible in the shipped code -- `IsA` at
0.5607 observed against 0.5335 under 200 pairing shuffles -- comes entirely from
that self-exclusion mask, an implementation artefact, and is roughly an eighth of
the 0.230 effect that was reported over the region-matched null.

So `relation_convergence.csv` never measured whether a relation is a reusable
transformation. It measures whether systems agree about the direction from the
source centroid to the target centroid. That is a real quantity and it stays in
the outputs, but it cannot support the Tier 2 criterion in
`RESULT_INTERPRETATION_PROTOCOL.md`, and run 001's relation headline should not
have been read as evidence about `IsA` as an operator.

A second consequence, found in the same check: the protocol asserted an ordering
of null difficulty rather than measuring one. For this statistic the region-matched
null is the *easiest* available, not the hardest. In run 001 `IsA` observed 0.561
against a global null mean of 0.554 (p = 0.507) and a region-matched null mean of
0.331. `IsA` failed the harder null and was reported on the easier one, which the
protocol's own rule forbids.

Decision, 2026-09-04:

1. `relation_pair_correspondence.csv` becomes the primary cross-system relation
   test. It keeps every pair's signature separate, builds
   `M[i][j] = cos(sig_A[i], sig_B[j])` over held-out anchors for each system pair,
   and tests the matched diagonal against the mismatched off-diagonal under a
   pairing-permutation null. Both multisets are held exactly fixed and only the
   correspondence moves, so the source and target sets alone cannot produce a
   result.
2. `relation_convergence.csv` is retained and relabelled descriptive. Its column
   meanings are unchanged; what changed is the claim it can carry.
3. Per-system-pair effects are written to
   `relation_pair_correspondence_by_system.csv`, so the Tier 2 requirement "not
   driven by one system pair" is checked rather than assumed.
4. Null difficulty is reported from the observed null means, never asserted in
   advance. Lead with whichever null actually ran harder.
5. `tests/test_pairing_sensitivity.py` locks all of it in: it asserts the retired
   statistic is pairing-invariant, that the new one is null on random geometry,
   and that it recovers a planted correspondence.

D16 items 1, 3 and 4 stand unchanged. This supersedes D16 item 2.

Generalisation worth carrying to the next design: with L2-normalised vectors,
`d(b,j) - d(a,j) = (a_hat - b_hat) . j_hat` exactly, so the "coordinate-free
relation signature" is the classical vector-offset analogy method expressed in an
anchor basis. It inherits that method's known failure modes, and the offset
literature belongs in `REFERENCES.md` before the relation claim is made again.
Before trusting any statistic that claims to test a pairing, mapping or
correspondence, permute that correspondence while holding everything else fixed
and confirm the number moves.

## D18 — The project states its own theory, and the handoff briefing is retired

Until now the only written statement of what "platonic semantic space" is taken
to mean lived in sections 3 and 4 of a 1,245-line coding-agent handoff briefing
whose headline was stale (it recorded that no pretrained model had been run,
which stopped being true on 2026-09-03). Everything else in that file duplicated
`EXPERIMENT.md`, `RESULT_INTERPRETATION_PROTOCOL.md` and the README.

The deeper problem it exposed: the project had an inductive half and no deductive
half. Comparing models only to each other measures agreement, and agreement has
no external referent, so no amount of analysis or visual design can make it into
a picture of a space. That ceiling is structural.

Decision, 2026-09-04:

1. `THEORY.md` is added and is the project's statement of what it is testing:
   platonic semantic space as a refinement lattice of distinguishability
   quotients under typed probes, rather than a metric space. The idea is
   generalised from the observation ladder in the separate `platonic-atlas-math`
   project, not from its mathematics.
2. The research agenda is stated as two ends that must meet — a deductive
   question about representation, and an inductive question about whether
   observed model convergence corresponds to it. A frame earns its place by
   making predictions the inductive side can check, not by being elegant.
3. Mathematics remains a separate project. `THEORY.md` §6 records why it is a
   *region* of the same lattice — formal, decidable probes with an independently
   checkable limit — and restates the standing rule that it is not the organising
   subject here.
4. The first prediction is recorded and already tested: if models are coarsenings
   of one structure their disagreements should nest rather than cross. They nest
   better than chance in 15 of 15 pairs, but only 6% to 23% of the way, rising
   with resolution. Reported in `THEORY.md` §7 and `FINDINGS_TO_DATE.md`.
5. Four handoff-era documents are removed: `CODING_AGENT_CONTEXT.md`,
   `CODING_AGENT_PROMPT.md`, `HANDOFF_ORIGIN.md` and `HANDOFF_MANIFEST.json`.
   They referenced only each other, nothing else in the repository referenced
   them, and their live content is now in `THEORY.md` §3 and §6. `PILOT_MANIFEST.json`
   goes with them: it recorded "no pretrained model has been run yet" and is
   superseded by `summary.json` and `EXPERIMENT_0_FREEZE.md`. Git history keeps
   all six.

This supersedes nothing in D1-D17. It adds the layer those decisions were
implicitly serving.
