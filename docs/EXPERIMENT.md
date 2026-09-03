# Semantic Relational Atlas — real pilot experiment

This is the experiment behind the proposed semantic-space atlas.

It is **not** a synthetic UI. When you run it, the semantic geometry is measured from hidden states of real language models.

## Pilot design

The benchmark contains:

- **212 matched concepts** in English and Simplified Chinese.
- **145 typed semantic relation probes** across 10 relation types.
- Three independently developed multilingual model families:
  - `Qwen/Qwen2.5-0.5B`
  - `bigscience/bloom-560m`
  - `facebook/xglm-564M`

The models are deliberately small so this can be run as a pilot before scaling.

## What the experiment asks

### 1. Does concept geometry survive language changes?

For every model and layer, the pipeline builds a representational dissimilarity matrix (RDM) over the *same concepts*.

English and Chinese can then be compared without requiring their raw hidden vectors to use the same coordinate system.

### 2. Do independently trained models converge?

The analysis selects one layer per model-language system by maximizing cross-system agreement of the ranked pairwise-distance geometry.

It then constructs a consensus RDM from the relationships that survive across systems.

### 3. Do reusable semantic transformations emerge?

The relation probe set includes:

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

For example:

```text
dog -> mammal
wolf -> mammal
whale -> mammal
```

and:

```text
fire -> smoke
exercise -> sweat
gravity -> fall
```

The analysis tests whether relation pairs have consistent transformations inside each system.

### 4. Coordinate-free relation signatures

Raw directions cannot be directly compared between unrelated neural networks.

For relation pair `a -> b`, define the signature:

```text
S(a,b)[j] = distance(b, anchor_j) - distance(a, anchor_j)
```

for every other concept `j`.

This asks: **how does the relationship to the rest of semantic space change when we move from a to b?**

Those coordinates are concept identities—not neuron identities—so the signature is directly comparable across models with different hidden sizes.

This is the experiment's most important addition over a simple embedding map.

## Run

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then:

```bash
python run_pilot.py
```

On Apple Silicon:

```bash
python run_pilot.py --device mps
```

On NVIDIA:

```bash
python run_pilot.py --device cuda
```

The first run downloads roughly three ~0.5B-parameter model families from Hugging Face and therefore needs internet access and several GB of disk space.

Models are loaded **one at a time**, so they do not all need to fit in RAM/VRAM simultaneously.

## Outputs

```text
outputs/
  representations/
    qwen25_05b__en__bare.npz
    qwen25_05b__zh__bare.npz
    ...
  analysis/
    system_alignment.csv
    cross_language_layer_curve.csv
    consensus_coordinates.csv
    relation_system_scores.csv
    relation_convergence.csv
    summary.json
  semantic_relational_report.html
```

### Read these first

`system_alignment.csv`
: Pairwise Spearman agreement between the selected high-dimensional geometries.

`cross_language_layer_curve.csv`
: Where English/Chinese convergence appears across model depth.

`consensus_coordinates.csv`
: 2-D coordinates for visualization **plus the more important per-concept stability metrics**.

`relation_system_scores.csv`
: Whether each labeled relation behaves like a reusable transformation inside each model/language system.

`relation_convergence.csv`
: Whether the coordinate-free transformation signature for a relation resembles itself across systems.

## Optional ConceptNet validation

The relation probe list is deliberately curated rather than generated from the models.

You can check which exact English edges are also present in ConceptNet:

```bash
python src/validate_with_conceptnet.py
```

This produces `outputs/conceptnet_validation.csv`.

ConceptNet remains an **external probe/overlay**. It never determines the neural coordinates.

## Statistical hardening now included

The pilot now treats the map as secondary and makes the following tests primary.

### Held-out layer selection

Concepts are deterministically split, stratified by broad semantic region:

- 60% **selection concepts**: used to choose the best layer in each model/language system.
- 40% **evaluation concepts**: never used for layer choice and used for headline convergence statistics.

This reduces the risk of choosing layers because they happen to fit the same concepts later used to claim convergence.

### System-geometry permutation null

For each pair of model/language systems, the observed held-out RDM Spearman correlation is compared against a null distribution produced by randomly permuting the concept identities in one system.

That preserves each model's internal geometry while destroying the proposed cross-system correspondence.

Output:

`outputs/analysis/system_alignment.csv`

Important columns:

- `heldout_spearman_rho`
- `null_mean`
- `effect_over_null`
- `z_vs_null`
- `permutation_p`
- `fdr_q`

### Neighborhood-stability null

The observed mean cross-system k-nearest-neighbor Jaccard agreement on held-out concepts is compared with systems whose concept identities have been independently relabeled.

Output:

`outputs/analysis/neighborhood_null_test.csv`

### Relation nulls

For every semantic relation, the relation signature is evaluated only on held-out concept-anchor dimensions. Sources and targets themselves can come from the full probe set, preserving statistical power.

Three nulls are used, and all of them preserve the multiplicity pattern of the observed targets:

1. **Global random target** - replacement targets drawn from anywhere in the benchmark.
2. **Region-matched random target** - each replacement drawn from the same broad semantic region as the target it replaces.
3. **Permuted pairing** - the observed source multiset and the observed target multiset are both kept exactly, and only which source goes with which target is destroyed. This is the strictest within-system baseline.

Multiplicity preservation is load-bearing. A null that resamples each pair's target independently destroys any target reuse present in the observed probes, which makes it strictly easier than the observed data and turns the test into a target-reuse detector: on pure random geometry the earlier version flagged a reuse-heavy `IsA` set at p <= .05 in 12 of 12 runs. `tests/test_null_calibration.py` locks this in, and the benchmark is built so every relation has all-distinct targets (with reuse, the correct null converges on the observed data and the test has no power at all).

The permuted-pairing null is deliberately not applied to cross-system mean-signature convergence, where the mean of `d(b, .) - d(a, .)` is very nearly invariant under a pairing permutation.

Two levels are reported:

1. within each model/language system:
   `relation_system_scores.csv`
2. coordinate-free convergence across systems:
   `relation_convergence.csv`

### Bootstrap uncertainty

- Global neighborhood stability receives a concept bootstrap CI.
- Cross-system relation convergence receives a relation-pair bootstrap CI.

### Multiple-testing correction

Permutation p-values are adjusted with Benjamini-Hochberg FDR within the relevant result family.

## Recommended first real run

For a fast diagnostic:

```bash
python run_pilot.py --device cuda --permutations 100 --bootstrap 200
```

For a more serious pilot:

```bash
python run_pilot.py --device cuda --permutations 1000 --bootstrap 2000
```

Use `--device mps` on Apple Silicon or omit `--device` to let the extractor choose.

## What would count as an interesting result?

Before looking at the data, the broad criterion should be:

> Held-out cross-model/cross-language semantic geometry and at least some labeled semantic transformations exceed identity-destroying and random-target nulls, with effects that remain after multiple-testing correction.

A pretty 2-D projection is not a success criterion.

## Still missing before publication-quality claims

- multiple prompt templates and prompt-averaged representations;
- larger concept coverage and contextualized word senses;
- translation ambiguity / lexical frequency controls;
- independent replication model set;
- stronger matched nulls beyond the included region-matched control (especially frequency/token-length matching);
- preregistered thresholds and analysis choices;
- bootstrap/permutation sensitivity analysis across random splits;
- propositions/events rather than only isolated lexical concepts.

## Why there is no whale data yet

The same machinery can later accept acoustic embeddings, but only after establishing shared non-linguistic anchors such as behavioral context, individual identity, conversational position, dive/foraging state, etc.

Putting whale calls directly into this human concept benchmark would imply semantic correspondences that have not been established.
