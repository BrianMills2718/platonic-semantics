# CODING AGENT CONTEXT — Shared / Platonic Semantic Space Project

## 0. Read this first

You are taking over an experimental research/code project whose goal is to test whether independently trained AI systems, operating across different human languages and eventually different modalities, recover **shared relational structure in semantic space**.

This is **not** a project to draw a pretty embedding map first.

This is **not** the separate "Platonic space of mathematical objects" project.

This is **not** a claim that whale communication has already been aligned with human or LLM semantics.

The immediate job is to produce and validate the first **real empirical results** from multiple multilingual LLMs, then build an atlas UI around whatever structure actually survives the tests.

The project's central question is:

> Do independently trained models, operating in different languages, recover some of the same coordinate-independent semantic neighborhoods and transformations?

A stronger long-term version is:

> Which relational structures remain invariant when independently learning systems encode the same underlying world?

The working term "Platonic semantic space" refers to a hypothesized shared statistical / relational structure. It is a hypothesis to test, not an assumed ground truth.

---

# 1. Origin of the project

The project began from several related observations / claims:

1. Multilingual LLMs can develop partially shared internal representations across languages.
2. Different neural networks can show increasingly alignable representational geometry.
3. The 2024 **Platonic Representation Hypothesis (PRH)** argues that increasingly capable models may converge toward a shared statistical representation of reality.
4. Separately, sperm-whale communication research has found contextual and combinatorial structure in vocalizations.
5. These ideas are often over-combined into a stronger claim such as "English, Chinese, independently trained AIs, and whales all use the same latent language." That stronger claim is **not established**.

The project should investigate the rigorous middle ground:
- shared geometry;
- shared neighborhoods;
- shared transformations / relations;
- how these change across model, language, layer, modality, and dataset.

---

# 2. Scientific position: what is established vs speculative

## Reasonably supported starting points

- Representation spaces can be compared through their **relational geometry** (kernels, distances, neighborhoods) without requiring their raw coordinates to match.
- Multilingual models can encode some cross-lingual abstractions.
- Some independently trained models exhibit representational similarities.
- These similarities may increase with capability in some settings.

## Important caveats

- PRH is a hypothesis / position, not a settled law.
- Representational similarity can be dataset-dependent.
- Training objectives and architecture/data choices matter.
- Shared geometry is not the same as identical internal vectors.
- A 2-D UMAP/MDS visualization is not evidence of universal structure.
- Multilingual similarity can be affected by translation pairs, shared training data, tokenization, lexical frequency, English dominance, and prompt construction.
- "Whale latent space = human latent space" has not been demonstrated.

The coding agent must preserve this epistemic discipline in both code and reporting.

---

# 3. The conceptual shift that defines the current project

The earliest prototype was essentially:

    models -> embeddings -> 2D projection -> cloud of dots

That is not sufficient.

Semantic relatedness is multi-relational. For example, `fire` has different relationships:

- similarity: fire <-> flame
- causality: fire -> smoke
- opposition/contextual contrast: fire <-> water
- physical relation: fire <-> heat
- function: fire -> cooking
- metaphor: fire <-> anger

One ordinary Euclidean distance cannot transparently express all of these simultaneously.

The current project therefore treats the target as a **latent relational atlas**, not one authoritative scatterplot.

The atlas should eventually expose:
- local semantic neighborhoods;
- relation/operator structure;
- cross-model stability;
- cross-language stability;
- layer dependence;
- disagreement / residuals;
- uncertainty.

The global 2-D projection is a display surface only.

---

# 4. Key lesson borrowed from the separate mathematics project

There was a separate project about mapping a "Platonic space of mathematical objects."

DO NOT turn this project into that.

Only retain these methodological lessons from it:

1. One object can be close to another under multiple distinct relation types.
2. The underlying object should be represented as a multi-relational structure rather than forced into one taxonomy.
3. A good atlas exposes multiple lenses / metrics.
4. Disagreement is information and should be visualized.
5. Local overlapping charts can be more faithful than one global map.

Mathematics may later be:
- one semantic region;
- a clean validation domain;
- a source of formally specified relations.

It is not the organizing subject of this project.

---

# 5. Current experimental hypothesis

## Primary hypothesis

Held-out semantic geometry will exhibit greater cross-model and cross-language agreement than correspondence-destroying null models.

## Relation hypothesis

At least some semantic relation types will produce reusable, coordinate-independent transformation signatures that converge across model/language systems more strongly than matched random relations.

Examples:

    dog -> mammal
    wolf -> mammal
    whale -> mammal

may instantiate a recurring `IsA`-like transformation.

Likewise:

    fire -> smoke
    exercise -> sweat
    gravity -> fall

may instantiate a recurring causal transformation.

The scientific question is whether those patterns survive:
- different model families;
- English vs Chinese;
- held-out anchors;
- permutation / matched nulls.

---

# 6. Current pilot benchmark

Source of truth:
- `benchmark/concepts.csv`
- `benchmark/relations.csv`

Current size:
- 212 concepts
- English + Simplified Chinese
- 145 curated relation pairs
- 10 relation types

Current relation types:
- `IsA`
- `PartOf`
- `HasA`
- `UsedFor`
- `Causes`
- `HasProperty`
- `Antonym`
- `SimilarTo`
- `AtLocation`
- `Associated`

These labels are **evaluation probes / interpretive labels**.
They do not define the neural geometry.

ConceptNet can optionally validate whether exact English edges exist externally, but ConceptNet must never position the concepts in the latent map.

---

# 7. Current real model set

Configured in `experiment_config.json`:

- `Qwen/Qwen2.5-0.5B`
- `bigscience/bloom-560m`
- `facebook/xglm-564M`

Languages:
- `en`
- `zh`

Why these models:
- small enough for a first pilot;
- different model families;
- multilingual;
- all support English and Chinese to useful degrees.

They are a **pilot set**, not proof of independence in a philosophical sense. They may share broad internet-era data distributions and similar Transformer ancestry.

---

# 8. Representation extraction

Implementation:
`src/extract_representations.py`

For each concept, language, model, and hidden layer:

1. construct a matched text stimulus;
2. tokenize;
3. run the causal LM with `output_hidden_states=True`;
4. pool non-padding, preferably non-special, token hidden states using mean pooling;
5. save:

    [concept, layer, hidden_dimension]

as compressed `.npz`.

Current prompt modes:

### bare

English:
    {term}

Chinese:
    {term}

### neutral

English:
    The concept is {term}.

Chinese:
    这个概念是{term}。

The current default is `bare`.

This is a major future control point: prompt form should later be averaged over multiple templates.

---

# 9. Why raw hidden vectors are NOT compared directly across models

Different models have:
- different hidden dimensionalities;
- arbitrary rotations / bases;
- different neuron identities;
- different layers.

Therefore do not compare neuron `k` in one network to neuron `k` in another.

Instead derive coordinate-free or alignment-invariant structures:
- pairwise representational dissimilarity matrices (RDMs);
- ranked distances;
- k-nearest-neighbor graphs;
- relation signatures defined over shared concept anchors;
- optionally later CKA / CKNNA / Procrustes-style measures as secondary analyses.

This is central to the project.

---

# 10. Representational distance geometry

For each model/language/layer, concepts produce hidden vectors `x_i`.

Current within-system distance:

    d(i,j) = cosine_distance(x_i, x_j)

This yields an RDM:

    D_s,l

where:
- `s` = system = model + language;
- `l` = layer.

Cross-system alignment is measured primarily with Spearman correlation between the upper triangles of RDMs.

The use of ranked geometry helps reduce sensitivity to raw scale.

---

# 11. Held-out layer-selection design

This is important and MUST NOT be casually removed.

Current analysis:
`src/analyze_semantic_geometry.py`

Concepts are stratified by broad region into:
- ~60% selection concepts
- ~40% evaluation concepts

Layers are selected using ONLY selection concepts.

The chosen layer for each model/language system maximizes agreement of its ranked RDM with the currently selected layers of the other systems through coordinate-ascent-like optimization.

Headline geometry tests are then performed on the held-out evaluation concepts.

Reason:
without this split, selecting a layer and evaluating it on the same concepts risks circularity / overfitting.

The display consensus may use all concepts only **after layers have been fixed**.

---

# 12. Consensus geometry

After layer selection:

1. calculate selected RDM for every model-language system;
2. rank-normalize the pairwise distances;
3. average those ranked distances;
4. reconstruct the consensus distance matrix.

This is the candidate shared semantic geometry.

Do NOT describe it as the unique true Platonic space.

It is:
> a consensus estimator over the included systems, stimulus set, pooling method, distance metric, and chosen analysis protocol.

2-D coordinates are generated only for display using classical MDS.

---

# 13. Neighborhood invariance

For each concept and system:
- compute k-nearest-neighbor set.

Across systems:
- compare these neighborhoods with Jaccard overlap.

Question:

> Does a concept retain approximately the same semantic neighborhood when model/language changes?

The primary held-out neighborhood statistic is compared against a null in which concept identities in systems are independently relabeled.

Relevant output:
`outputs/analysis/neighborhood_null_test.csv`

---

# 14. Coordinate-free semantic relation signatures

This is the most important conceptual addition beyond ordinary embedding maps.

Raw displacement:

    x_b - x_a

cannot be compared across unrelated neural coordinate systems.

Instead define, for relation pair `a -> b`, a vector indexed by shared concept anchors `j`:

    S(a,b)[j] = d(b,j) - d(a,j)

Interpretation:

> How does our relationship to the rest of semantic space change when moving from concept a to concept b?

The dimensions are **concept identities**, not neurons.

Therefore the signature can be compared directly across model families with different hidden dimensions.

Example:

    dog -> mammal

produces a pattern of changes in distance to:
- cat
- wolf
- animal
- tree
- car
- justice
- etc.

If `wolf -> mammal` and analogous relations produce similar patterns, a reusable relational transformation may exist.

If the mean `IsA` signature is also similar in Qwen, BLOOM, and XGLM, that is evidence for cross-system relational convergence.

---

# 15. Relation evaluation and its current compromise

The current small pilot needs enough relation examples per type.

Therefore:
- relation source/target pairs may use the full curated relation set;
- the **signature comparison dimensions** are restricted to held-out concept anchors.

This gives a partially held-out relational test while retaining relation-pair power.

This is a pragmatic pilot compromise, not the final publication-grade design.

Future versions should include:
- explicit train/eval relation-pair splits;
- larger relation datasets;
- matched lexical and frequency controls;
- preregistered relation selection.

Do not overstate the current relation split.

---

# 16. Statistical hardening already implemented

The hardened pipeline includes:

## A. System geometry permutation null

For each pair of selected systems:
- keep one RDM intact;
- permute concept identity in the other;
- compare observed held-out Spearman correlation to the null.

Outputs include:
- heldout Spearman rho
- null mean
- null SD
- effect over null
- z vs null
- permutation p
- FDR q
- descriptive bootstrap CI

File:
`system_alignment.csv`

## B. Neighborhood permutation null

Cross-system held-out kNN Jaccard is compared against independently relabeled systems.

File:
`neighborhood_null_test.csv`

## C. Relation random-target null

Keep relation sources but replace targets with random concepts.

## D. Harder region-matched relation null

Replace the target with another concept from the same broad semantic region where possible.

This reduces trivial success caused by coarse category structure.

## E. Bootstrap uncertainty

Current use:
- concept bootstrap for neighborhood stability;
- relation-pair bootstrap for cross-system relation convergence;
- descriptive bootstrap for RDM correlations.

## F. Multiple-testing correction

Benjamini-Hochberg FDR correction is applied within result families.

Preserve these unless there is a documented statistical reason to change them.

---

# 17. Primary outputs

After a real run, inspect these FIRST.

## `system_alignment.csv`

Question:
> Do held-out high-dimensional semantic geometries align across systems above the identity-shuffling null?

Key columns:
- `system_a`
- `system_b`
- `heldout_spearman_rho`
- `null_mean`
- `effect_over_null`
- `z_vs_null`
- `permutation_p`
- `fdr_q`

## `neighborhood_null_test.csv`

Question:
> Do models agree on held-out local semantic neighborhoods above chance?

## `cross_language_layer_curve.csv`

Question:
> Where in depth does English/Chinese geometry become most similar?

Potentially interesting pattern to test, NOT assume:

    early layers -> more language/form specific
    middle layers -> stronger shared concept geometry
    late layers -> output/model specific

## `relation_system_scores.csv`

Question:
> Does each labeled relation behave consistently as a transformation inside each model/language system?

Includes global and region-matched target nulls.

## `relation_convergence.csv`

Question:
> Does a coordinate-free relation signature resemble itself across different systems?

This is one of the project's most important result files.

## `consensus_coordinates.csv`

Contains:
- display x/y;
- global stability;
- model stability;
- language stability;
- concept metadata.

The x/y values are visualization aids, not primary evidence.

## `summary.json`

Run metadata and selected layers.

## `semantic_relational_report.html`

Human-facing overview of the main results.

---

# 18. What counts as success

Do not use "the map looks interesting" as a success criterion.

A useful pilot result would satisfy at least some of the following:

### Geometry gate
Held-out cross-system RDM alignment is:
- positive;
- meaningfully above permutation null;
- robust across multiple system pairs;
- preferably survives FDR.

### Neighborhood gate
Held-out kNN agreement is:
- above the identity-destroying null;
- nontrivial in effect size.

### Relation gate
At least some relation types show:
- positive within-system consistency above null;
- cross-system relation-signature convergence;
- positive effect over the harder region-matched null;
- FDR-adjusted evidence.

### Robustness gate
Effects survive at least:
- both bare and neutral prompts;
- several random train/eval splits;
- reasonable k values / distance variants.

Strong evidence should replicate in a second model set.

---

# 19. What counts as failure

Failure is scientifically useful.

If results are weak or vanish under nulls:
- do not force a map;
- do not cherry-pick model pairs;
- do not invent a "Platonic" narrative.

Diagnose:
- isolated-word ambiguity;
- bad translations;
- tokenization differences;
- lexical frequency;
- model size too small;
- poor pooling;
- layer-selection instability;
- relation labels too coarse;
- metric choice;
- training-data overlap / differences;
- insufficient benchmark size.

A failed pilot should produce a clear diagnostic report and revised experiment.

---

# 20. Current known limitations

Treat these as active engineering/research issues.

1. **Isolated concepts are ambiguous.**
   - `bank`, `light`, etc. can have multiple senses.
   - current benchmark partly avoids severe ambiguity, but not fully.

2. **Translations need stronger validation.**
   - English and Chinese terms should refer to the same intended sense.
   - later add bilingual review / sense IDs.

3. **Prompt dependence.**
   - current default uses bare words.
   - later average multiple matched prompt formulations.

4. **Pooling dependence.**
   - mean token pooling is convenient but not uniquely justified.
   - compare last-token, mean, weighted/token-specific strategies.

5. **Tokenization confounds.**
   - record token counts and potentially control for them.

6. **Frequency confounds.**
   - common concepts may have more stable representations.
   - add frequency-matched nulls.

7. **Small model set.**
   - three ~0.5B families are a pilot only.

8. **Model "independence" is limited.**
   - they share broad Transformer conventions and may share internet data distributions.

9. **Dataset dependence.**
   - PRH-related criticism shows representational similarity can depend strongly on stimulus dataset and training objective.

10. **Relation evaluation is only partly held out.**
    - held-out anchors are used, but source/target pairs may come from full probe list.

11. **2-D projection loses information.**
    - never use visual proximity as primary statistical evidence.

12. **No real model results have been produced in the original ChatGPT environment yet.**
    - synthetic tensors were used only to smoke-test code paths.

---

# 21. Real-run status

The original execution environment could not perform the real-weight experiment because:
- CPU-only PyTorch was exposed;
- model download hosts were unavailable from the container;
- model runtime dependencies were not locally installed;
- no connected GPU inference service was available.

This is why the package contains:

`notebooks/semantic_relational_atlas_real_run.ipynb`

That notebook is the current preferred execution handoff.

It:
1. checks for a Colab GPU;
2. installs dependencies;
3. unpacks the complete experiment embedded inside itself;
4. downloads the three model checkpoints;
5. extracts English and Chinese hidden states;
6. runs hardened statistics;
7. generates the HTML report;
8. displays headline result tables;
9. applies a conservative automatic interpretation;
10. bundles the real outputs for download.

Run on a T4 or better if possible.

---

# 22. Immediate coding-agent priorities

Execute in this order.

## Priority 1 — make the real run succeed

Run the notebook or project on a GPU environment with internet access.

If dependencies break:
- fix compatibility;
- preserve experiment semantics;
- record exact versions.

Do NOT modify the statistical design merely to make the run easier.

Capture:
- Python version;
- PyTorch version;
- Transformers version;
- tokenizer versions;
- exact model revisions / commit hashes if possible;
- GPU;
- random seed;
- wall-clock/runtime;
- peak memory if practical.

## Priority 2 — validate extracted representations

Before analysis, verify:
- correct number of concepts in every system;
- no NaN/Inf representations;
- layer counts make sense;
- hidden dimensions match model configs;
- EN/ZH concept ordering is identical;
- padding/special token masking works;
- tokenizer actually supports requested language text;
- representation files are not accidentally duplicated.

Generate a machine-readable QC report.

## Priority 3 — run quick statistics

Use something like:
- 100 permutations;
- 200 bootstraps.

This is only to catch errors.

## Priority 4 — run serious pilot

Use:
- 1000+ permutations;
- 2000+ bootstrap replicates.

Persist all raw outputs.

## Priority 5 — inspect null-corrected results

Do not begin with 2-D coordinates.

First examine:
1. `system_alignment.csv`
2. `neighborhood_null_test.csv`
3. `relation_system_scores.csv`
4. `relation_convergence.csv`
5. `cross_language_layer_curve.csv`

## Priority 6 — robustness suite

If there is a signal:
- bare vs neutral prompt;
- multiple split seeds;
- k = 5, 10, 20;
- cosine vs correlation distance if justified;
- mean vs last-token pooling;
- relation null sensitivity.

## Priority 7 — only then update the atlas UI

Use actual results.

---

# 23. Desired next-generation experiment after the pilot

If Experiment 0 passes:

## Scale concepts
212 -> 500 -> 2,000+

## Add contextualized senses

Instead of only:

    bank

use:

    bank_financial:
      "She deposited money at the bank."

    bank_river:
      "They sat on the bank of the river."

Matched Chinese examples should express the same senses.

## Add propositions / events

Examples:

English:
    The dog chased the cat.
Chinese:
    狗追赶了猫。

English:
    Heavy rain caused flooding.
Chinese:
    暴雨导致了洪水。

English:
    A wheel is part of a bicycle.
Chinese:
    车轮是自行车的一部分。

This moves the project from lexical semantics toward structured situations.

## Add languages
Suggested next:
- Spanish
- Arabic
- Japanese
- perhaps typologically distant / different-script languages

## Add model families
Aim for:
- different labs;
- different tokenizer families;
- different scale;
- base rather than instruction-tuned where possible for cleaner representation comparison.

## Add model size axis
Test whether shared geometry strengthens with model capability/scale.

## Add modalities
After text results are established:
- images of matched concepts/events;
- audio/environmental sound;
- diagrams;
- video/event representations.

---

# 24. Unsupervised relation discovery — important future direction

The current relation labels are supervised probes.

A more ambitious phase should ask whether relation/operator structure can be **discovered without naming it first**.

Potential approach:

1. sample many concept pairs;
2. compute coordinate-free signatures;
3. embed/cluster the signatures;
4. test whether clusters recur across model/language systems;
5. only then inspect whether a cluster corresponds to:
   - IsA
   - cause
   - part-whole
   - opposition
   - temporal transition
   - agency
   - spatial relation
   - metaphor
   - something not already in our ontology.

This moves toward discovering an **algebra of meaning** rather than merely validating ConceptNet categories.

Do not let external ontologies predetermine all possible relations.

---

# 25. Atlas / UI direction after real data exists

The eventual UI should NOT primarily be one giant scatterplot.

Preferred interaction model:

## A. Concept inspector

Example: `fire`

Show:
- consensus nearest neighbors;
- selected model nearest neighbors;
- English vs Chinese neighborhood stability;
- model-by-language stability matrix;
- uncertainty;
- relation memberships.

## B. Relation/operator explorer

Example: `Causes`

Show:
- relation pairs;
- within-system consistency;
- cross-system signature similarity;
- matched-null baseline;
- bootstrap CI;
- which model/language systems support or break the relation.

## C. Layer-depth view

Show:
- EN/ZH convergence by relative layer;
- cross-model convergence by layer;
- selected layers;
- uncertainty across prompts/splits.

## D. Disagreement view

Highlight:
- concepts with unstable neighborhoods;
- relation types with weak or negative convergence;
- language-specific distortions;
- model-family-specific structure.

## E. Local maps

Allow "map neighborhood" around a selected concept or relation.

A local atlas is preferable to pretending a single 2-D projection preserves all semantics.

## F. Global view

Can exist for orientation, but always:
- label it as a projection;
- expose stress/eigenvalue capture;
- show confidence/stability;
- allow switching metrics.

The existing synthetic prototype is included under `prototypes/` only as UI inspiration.

Never present its synthetic coordinates as empirical output.

---

# 26. External semantic graphs: correct role

Useful sources may include:
- ConceptNet
- WordNet
- Wikidata
- domain ontologies
- formal mathematical graphs

Their role:
- labels;
- overlays;
- probe generation;
- validation;
- relation naming;
- external comparison.

Their role is NOT:
- deciding the latent coordinates;
- defining the "true" semantic geometry.

The neural geometry should emerge from model representations.

---

# 27. Whale communication: future extension, not current evidence

The project was partly motivated by discussion of animal communication, especially sperm whales.

Current scientific position:

- sperm whale codas show contextual and combinatorial structure;
- machine learning can learn useful acoustic representations;
- there is no established English/Chinese/LLM-to-whale semantic coordinate map;
- do not place whale calls next to English concepts just because both have embeddings.

A defensible future extension would anchor whale representations to non-linguistic shared variables such as:
- individual identity;
- social group;
- turn position;
- response timing;
- approach/contact behavior;
- surface/dive state;
- foraging/social context;
- environmental context.

Then test whether relational geometry associated with those contexts aligns across:
- acoustic representations;
- behavioral/event representations;
- human-language descriptions of the same observed contexts.

The question is NOT:
> Which whale click means "fish"?

The question is:
> Do independently learned relational structures over shared observable contexts show invariant geometry?

Do not work on this until the text-only experiment is credible.

---

# 28. PRH / WIT precursor

An earlier prototype used public features released with the Platonic Representation Hypothesis over shared WIT image/text stimuli.

It built:
- model RDMs;
- cross-model layer matching;
- rank-normalized consensus distances;
- classical MDS;
- neighborhood agreement.

That code is retained under:

`archive/prh_wit_baseline/`

It is useful as:
- methodological precedent;
- a separate baseline;
- evidence that the project originally followed PRH's distance-geometry idea.

It is NOT the current main experiment because the current project focuses on explicit semantic concepts and relations.

---

# 29. Reproducibility rules for the coding agent

For every real experiment:

1. Do not overwrite previous runs.
2. Create a run directory with timestamp or semantic version.
3. Save exact config JSON.
4. Save model IDs and revision hashes where possible.
5. Save package versions.
6. Save random seeds.
7. Save concept/relation benchmark snapshot.
8. Save QC report.
9. Save raw representation metadata.
10. Save all statistical outputs.
11. Save report HTML.
12. Save a concise `INTERPRETATION.md` that distinguishes:
    - observed data;
    - statistical inference;
    - speculation.

Suggested run structure:

    runs/
      2026-09-xx_pilot_v1/
        environment.json
        config.json
        benchmark/
        qc/
        representations/
        analysis/
        report.html
        INTERPRETATION.md

---

# 30. Coding style / project discipline

- Prefer simple inspectable NumPy/Pandas/SciPy code for core statistics.
- Keep scientific calculations separate from visualization.
- Functions should have explicit inputs/outputs.
- Avoid hidden global state.
- Add unit tests for:
  - RDM symmetry;
  - diagonal zeros;
  - permutation behavior;
  - split reproducibility;
  - signature dimensions;
  - FDR correction;
  - concept ordering;
  - null generation constraints.
- Add integration tests on tiny synthetic data.
- Mark synthetic tests loudly.
- Do not silently catch errors that can invalidate the analysis.
- Validate all inputs before calculating statistics.
- Use deterministic seeds where possible.
- Log enough metadata to reproduce results.

---

# 31. Suggested engineering improvements before scaling

These are not required to get the first real run, but should be queued.

## Extraction
- add `--model-revision`;
- save tokenizer statistics per concept/language;
- save pooling method;
- support last-token / mean / configurable pooling;
- support several prompts per concept;
- batch prompt templates;
- record dtype and device;
- support resume/caching.

## Analysis
- support multiple split seeds;
- aggregate split-level effect sizes;
- stronger paired/bootstrap methods for RDM correlation;
- frequency/token-length-matched nulls;
- explicit relation-pair train/eval split;
- compare several distance metrics;
- optional CKA/CKNNA;
- stability across stimulus subsets;
- test model-size trends.

## Benchmark
- canonical sense IDs;
- translation review;
- POS fields;
- lexical frequency fields;
- token length fields;
- relation provenance;
- relation confidence;
- negative examples;
- contextualized examples.

## Reporting
- result-card style summary with pass/fail gates;
- explicit null distributions;
- uncertainty plots;
- layer curves;
- prompt/split sensitivity;
- no claim from 2-D plot alone.

---

# 32. High-level roadmap

## Experiment 0 — Does shared semantic geometry exist?
CURRENT PHASE.

Real models:
- Qwen 0.5B
- BLOOM 560M
- XGLM 564M

Languages:
- English
- Chinese

Outputs:
- geometry alignment;
- neighborhood invariance;
- relation convergence;
- null tests.

## Experiment 0.5 — Robustness
- multiple prompt templates;
- multiple splits;
- pooling variants;
- k sensitivity;
- matched nulls.

## Experiment 1 — Larger lexical/concept atlas
- 500–2,000 concepts;
- more relation instances;
- more languages;
- more model families.

## Experiment 2 — Contextual semantics
- disambiguated senses;
- propositions;
- events;
- roles;
- causality;
- spatial/temporal structure.

## Experiment 3 — Unsupervised relational algebra
- discover recurring relation signatures without labels.

## Experiment 4 — Multimodal
- images;
- sound;
- video;
- diagrams.

## Experiment 5 — Non-human communication
Only with defensible shared behavioral/context anchors.

---

# 33. Current "do not do" list

Do NOT:

- build a map of mathematical objects as the main product;
- assume "Platonic space" exists;
- treat PRH as proven;
- compare raw hidden vectors across unrelated models as if neuron coordinates align;
- use the synthetic UI data as results;
- use synthetic smoke-test tensors as results;
- start with UMAP and infer science from clusters;
- cherry-pick layers on evaluation concepts;
- use ConceptNet as ground-truth coordinates;
- call whale communication "the same latent language";
- manufacture semantic translations for whale calls;
- suppress negative/null results;
- optimize the benchmark after inspecting headline results without documenting that exploration.

---

# 34. The coding agent's immediate task statement

Your next job is:

> Run and validate the real multilingual pilot on Qwen2.5-0.5B, BLOOM-560M, and XGLM-564M; preserve the held-out and null-test design; produce reproducible real outputs; diagnose any failures; and only after that connect those outputs to a data-driven semantic atlas.

First deliverables:

1. successful real extraction for all 6 model-language systems;
2. QC report;
3. quick analysis run;
4. full permutation/bootstrap run;
5. headline result tables;
6. conservative interpretation;
7. identified methodological weaknesses;
8. only if signal is real, updated atlas UI using real outputs.

If the pilot is negative, deliver the negative result clearly and propose the next discriminating experiment.

---

# 35. Reference reading

See `docs/REFERENCES.md`.

At minimum, understand:
- Huh et al. 2024 PRH;
- Wendler et al. 2024 multilingual latent language;
- Brinkmann et al. 2025 shared multilingual grammatical features;
- Ciernik et al. 2025 dataset/objective dependence critique;
- ConceptNet relation definitions;
- the three model cards;
- sperm-whale combinatorial-structure paper only as future context.

---

# 36. Final project principle

The project is not trying to prove that all minds secretly speak one language.

It is trying to measure:

> which relationships among meanings remain stable when the encoding system changes.

If stable structures exist, the atlas should reveal them.

If they do not, the atlas should reveal the disagreement.
