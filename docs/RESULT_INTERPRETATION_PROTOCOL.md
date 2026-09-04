# RESULT INTERPRETATION PROTOCOL — BEFORE FIRST REAL RUN

This protocol is prospective relative to the first real pretrained-model outputs. It is not a formal preregistration.

## Preferred terminology

Use:
- shared semantic geometry;
- cross-system representational convergence;
- stable semantic neighborhoods;
- relation-signature convergence.

Avoid saying the pilot has discovered:
- the universal Platonic space;
- one language-independent internal language;
- a whale-human semantic space.

## Result tiers

### Tier 0 — No convincing signal
Geometry/neighborhood/relation results fail appropriate nulls.

Interpretation:
> This pilot does not provide evidence for shared semantic relational geometry under the tested design.

### Tier 1 — Geometry/neighborhood signal only
Held-out geometry or neighborhoods beat null, but relation operators do not survive harder controls.

Interpretation:
> Systems share some concept organization, but reusable cross-system semantic transformations are not established.

### Tier 2 — Some relation convergence
At least some relation types, **in `relation_pair_correspondence.csv`**:
- show matched pair signatures beating mismatched ones across systems;
- exceed the pairing-permutation null;
- survive FDR;
- are not driven by one system pair — check
  `relation_pair_correspondence_by_system.csv`, do not assume it.

A result in `relation_convergence.csv` cannot reach this tier no matter how
large, because that statistic does not depend on the pairing. Run 001 was scored
against the old wording and reached Tier 1; under this wording Tier 2 was not
merely unmet, it was untestable.

Interpretation:
> Some labeled semantic relations exhibit coordinate-independent transformation structure recurring across the tested systems.

### Tier 3 — Robust pilot
Tier 2 plus robustness to:
- bare vs neutral prompts;
- multiple split seeds;
- k choices;
- pooling choices;
- obvious tokenization artifacts.

Interpretation:
> Robust evidence of shared relational semantic structure across the tested systems.

Still do not call it universal.

### Tier 4 — Replicated phenomenon
Requires:
- independent model-family replication;
- larger benchmark;
- contextualized stimuli;
- additional languages;
- preferably a held-out model family.

Only here should stronger "Platonic semantic structure" language be considered.

## Primary result order

1. system_alignment.csv
2. neighborhood_null_test.csv
3. relation_convergence.csv
4. relation_system_scores.csv
5. cross_language_layer_curve.csv
6. 2-D map last

## Relation rule

**Amended 2026-09-04 (D17).** The original rule asserted an ordering of null
difficulty instead of measuring one, and named a cross-system statistic that
cannot test what this protocol asks of it. Both are corrected below.

The primary cross-system relation test is `relation_pair_correspondence.csv`:
matched versus mismatched pair signatures under a pairing-permutation null. Only
it can support Tier 2, because only it holds the source and target multisets
fixed and moves the correspondence.

`relation_convergence.csv` is **descriptive**. Its statistic is algebraically
invariant to which source is paired with which target, so it measures agreement
about the source-centroid-to-target-centroid direction, not a reusable
transformation. Never cite it as relation evidence.

Lead with the hardest null the result survives, and say which one it was.
**Difficulty is read off the observed null means in that run, never assumed.**
For within-system consistency the order has held as `permuted_pairing_*`
(strictest: both observed multisets exact) > `region_matched_*` > `global_*`. For
the descriptive cross-system statistic it is inverted -- in run 001 the global
null ran harder than the region-matched one for every relation -- which is how
`IsA` came to be reported on a null it beat while failing the one it did not.

Never report the easier null's result when the harder one fails. Check which was
which by comparing `*_null_mean`, before writing the sentence.

Also read `target_reuse_ratio` in the relation outputs before interpreting any
relation. It must be 1.00; anything higher means that relation's probes reuse a
target and its numbers should not be interpreted at all.

## Effect size rule

Always report:
- observed statistic;
- null mean;
- effect over null;
- uncertainty;
- p-value;
- FDR q-value.

Do not equate statistical significance with scientific importance.

## Mixed results are meaningful

Example:
- IsA robust;
- PartOf robust;
- Causes weak;
- Antonym unstable;
- Associated null.

That would suggest a **structure of invariance**, not a failed universal-all-relations hypothesis.

## No benchmark editing after headline outputs

Any post-result benchmark change must produce a new version and be labeled accordingly.

## Atlas rule

Once real outputs exist, the atlas may display those quantities.

It must expose:
- uncertainty;
- disagreement;
- null/effect context;
- projection limitations.

Synthetic/demo data must remain visibly labeled.
