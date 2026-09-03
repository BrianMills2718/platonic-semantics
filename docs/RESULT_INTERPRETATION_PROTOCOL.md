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
At least some relation types:
- show positive cross-system similarity;
- exceed the region-matched null;
- survive FDR;
- are not driven by one system pair.

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

Lead with the hardest null the result survives, and say which one it was. For
within-system consistency that order is
`permuted_pairing_*` (strictest: both observed multisets exact) >
`region_matched_*` > `global_*`. For cross-system convergence the permuted-pairing
null does not apply, so `region_matched_*` is the hardest available.

Never report the easier null's result when the harder one fails.

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
