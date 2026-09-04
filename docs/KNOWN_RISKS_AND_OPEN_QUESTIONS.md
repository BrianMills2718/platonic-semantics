# KNOWN CODE RISKS, SCIENTIFIC RISKS, AND OPEN QUESTIONS

This is a review checklist. Items are potential risks, not claims that every item is currently a bug.

**Status note, 2026-09-03.** Items 8, 9 and 16 all circle the relation null, and
none of them named the defect that was actually there: the null resampled each
pair's target independently and so destroyed the target reuse the probes have,
making it a target-reuse detector. It was found by running the pipeline on random
geometry, not by reviewing it. See `FINDINGS_TO_DATE.md` and D16 in
`DECISION_LOG.md`. Items marked **[addressed]** below have been fixed; the rest
stand.

## Highest priority checks

### 1. Layer-selection stability
Current selection is iterative and may depend on initialization.

Check:
- multiple initializations;
- selected relative depth;
- headline sensitivity to selected layer.

### 2. Layer indexing
`hidden_states` usually includes embedding output plus block outputs.

Record both hidden-state array index and transformer-block interpretation.

### 3. Mean pooling
Current method mean-pools content tokens.

Test:
- mean pooling;
- final content token;
- other justified pooling variants.

### 4. Prompt dependence
Primary stimuli are bare lexical items.

Test neutral and multiple contextual templates before strong claims.

### 5. Float16 storage
Representations are stored compressed in float16.

Check one subset in float32 to quantify geometry changes.

### 6. Consensus weighting
Systems are equally weighted. They are not statistically independent.

Later compare system-, model-, and language-balanced consensus schemes.

### 7. Relation-signature normalization
Current signature uses raw cosine-distance changes and compares centered signature cosine.

Test prospectively:
- ranked distance-change signatures;
- standardized RDMs;
- z-scored anchor-distance changes.

### 8. Relation holdout is incomplete
Relation source/target pairs may use the full benchmark while anchor dimensions are held out.

A larger benchmark needs explicit relation-pair train/eval splits. Still open.

### 9. Region-matched null is coarse — **[partly addressed]**
Broad regions are assistant-curated.

All relation nulls now preserve the multiplicity pattern of the observed targets,
and a permuted-pairing null keeps both observed multisets exactly. The coarseness
of the *regions themselves* is unchanged.

Future matching should include:
- semantic region;
- POS;
- token length;
- lexical frequency;
- baseline source-target similarity.

### 10. RDM bootstrap dependence
Pairwise distances are not independent.

Treat the current RDM bootstrap CI as descriptive; permutation inference is primary.

### 11. kNN depends on k
Report sensitivity at k = 5, 10, 20 or comparable values.

### 12. "Independent models" are only partially independent
The models are separate projects but share Transformer ancestry and may have overlapping internet-era data distributions.

### 13. Pilot models are small
Weak effects may reflect scale rather than absence of the phenomenon.

### 14. Translation equivalence is not guaranteed
Bare English/Chinese lexical items can differ in sense boundaries and tokenization.

### 15. Relation MRR can punish legitimate alternate targets
Many semantic relations have multiple valid targets.

### 16. Relation labels are heterogeneous
Associated is broad/noisy.
Antonym/SimilarTo are often symmetric.
PartOf/HasA are inverses.
IsA may mix class-level intuitions.

`IsA` was additionally the most target-reusing relation (27 pairs, 9 targets),
which is what made the null defect visible there first. The probe set now
enforces all-distinct targets per relation, so heterogeneity remains a risk but
reuse no longer compounds it.

### 19. A null can be easier than the data — **[addressed, keep the check]**
Any null that resamples a component independently can destroy structure the
observed data has, and thereby become easier than the data rather than harder.

`tests/test_null_calibration.py` runs the analysis on pure random geometry and
fails if it reports a result. Run it after touching any null, and prefer it to
re-reading the design: the design was read carefully several times without
catching this.

### 20. The gate is enforced only for relations
Geometry and neighborhood nulls permute concept identity, which does not have
the same structural-destruction problem. They have not been calibrated on random
geometry as carefully as the relation path. Worth doing before headline claims.

### 17. No formal preregistration
The interpretation protocol is prospective relative to the first real run, not a registered study.

### 18. Dataset dependence is a core risk
Published representational convergence does not guarantee convergence on this stimulus set.

### 21. The deep layers are nearly degenerate — **[open, not addressed]**
The load-bearing sensitivity run selects layer 24 for five of six systems. At that
layer the 212 concepts have a mean pairwise cosine of 0.714 (qwen|en), 0.791
(qwen|zh), 0.924 (xglm|en), 0.944 (xglm|zh), 0.978 (bloom|en) and **0.997**
(bloom|zh) — against 0.037-0.415 at layer 0. Rank orderings are being read from
the last fraction of a percent of a collapsed space. Either justify measuring
there or whiten before doing so.

### 22. The headline is not invariant to centering — **[open, not addressed]**
Nothing preregistered whether representations are mean-centred before the cosine
RDM, and the choice moves results materially. Diagnostics over all 212 concepts:
mean-centering moves qwen|zh-bloom|zh from 0.453 to 0.607, while removing the top
principal component moves bloom|en-xglm|zh from 0.162 to 0.021 and
bloom|en-xglm|en from 0.250 to 0.106 — and moves qwen|en-bloom|en *up*, 0.311 to
0.427. The top PC in LM space typically carries frequency and length, which is
the confound the lexical control targets. Fix the choice deliberately and report
under it; these figures are directional, not corrected effect sizes.

### 23. The `IsA` null matches region, not taxonomic level — **[open]**
All 18 `IsA` targets are superordinate category nouns (`mammal`, `tool`,
`emotion`, `metal`). The region-matched null draws replacements from the same
broad region, which is mostly basic-level terms, so abstractness is not
controlled. Any surviving `IsA` result needs a null matched on taxonomic level.

### 24. `corr()` still maps a non-finite result to 0.0 — **[open]**
`src/analyze_semantic_geometry.py` returns `0.0` rather than raising when
Spearman is non-finite. This is the silent zero that hid the bloom-560m float16
NaN in run 001; the cause was fixed, the swallow was not. It reads downstream as
"these systems do not agree", which is a wrong answer wearing the shape of a
right one.

### 25. Layout in the atlas is a display convenience — **[bounded]**
`results/run_001/atlas.html` positions concepts by force-directed layout over the
agreement graph. That is one of many valid arrangements; the links are the data
and the coordinates are not. Positive control on record: same-region concepts
land at mean layout distance 0.33 against 0.73 for unrelated pairs, so the layout
is not noise. The map code is display built on run outputs and is not under test.

### 26. Prompt reliability caps the headline, and one system fails it — **[open]**
`scripts/prompt_reliability_ceiling.py` measures how much of a system's geometry
survives rewording the stimulus. `bloom_560m|en` scores 0.156 and appears in 5 of
the 15 pairs behind the headline; it also normalises above 1.0 against
`qwen25_05b|en`, meaning it agrees with another model more than with itself, which
voids that pair. XGLM has no second extraction yet. A training-seed ceiling
(Pythia publishes same-architecture seeds) remains the better measurement and is
not done.

## Open questions

1. Are local neighborhoods more reproducible than global RDM geometry?
2. Is CKA/CKNNA more stable than RDM Spearman for some comparisons?
3. Do relation signatures add information beyond coarse semantic category?
4. Does language invariance peak at a characteristic relative depth?
5. Does convergence increase with model scale?
6. Are relation signatures better represented as vectors, local operators, or low-rank transformations?
7. Can relation families be discovered without labels?
8. Can consensus from some systems predict a held-out model family?
9. How should an atlas show uncertainty without making a 2-D manifold look more real than it is?
