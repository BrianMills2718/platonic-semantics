# platonic-semantics

Do independently trained language models, in different languages, arrive at the
same *shape* of semantic space — and do labelled semantic relations behave like
reusable transformations inside it?

This repository is the experiment, not an answer. **No pretrained model has been
run yet.** Every result file the pipeline can produce is hypothetical until the
first real run happens.

Part of a wider set of "platonic" projects; the mathematics side lives in
[`platonic-atlas-math`](https://github.com/BrianMills2718/platonic-atlas-math).

## The idea

Two models with different hidden sizes have no shared coordinate system, so
their vectors cannot be compared directly. Two things here can be:

**Geometry.** For each model and language, build a representational
dissimilarity matrix (RDM) over the *same* concepts. The RDM is indexed by
concept identity, not by neuron, so English and Chinese and Qwen and BLOOM can
be compared by how similarly they rank the same pairs.

**Relation signatures.** For a relation pair `a -> b`, define

```text
S(a,b)[j] = distance(b, anchor_j) - distance(a, anchor_j)
```

over every other concept `j`: how does the relationship to the rest of semantic
space change when you move from `a` to `b`? Those coordinates are concept
identities, so the signature is directly comparable across models that share
nothing else.

## Benchmark

212 concepts matched in English and Simplified Chinese, 145 relation probes
across 10 types (`IsA`, `PartOf`, `HasA`, `UsedFor`, `Causes`, `HasProperty`,
`Antonym`, `SimilarTo`, `AtLocation`, `Associated`), and three independently
developed multilingual families at ~0.5B parameters: `Qwen/Qwen2.5-0.5B`,
`bigscience/bloom-560m`, `facebook/xglm-564M`.

**Every relation type has all-distinct targets, and CI enforces it.** This is a
hard design constraint, not tidiness — see below.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pilot.py --device cuda --permutations 1000 --bootstrap 2000
```

`--device mps` on Apple Silicon; omit `--device` to autodetect. The first run
downloads three ~0.5B models, one at a time, so they never need to fit in
memory together. `notebooks/semantic_relational_atlas_real_run.ipynb` is a
self-contained Colab version.

Outputs land in `outputs/analysis/` (gitignored). Read `system_alignment.csv`,
`neighborhood_null_test.csv`, `relation_system_scores.csv` and
`relation_convergence.csv`. `docs/EXPERIMENT.md` describes each column.

## What this repository gets right, and what it does not

### The null has to preserve what it is not testing

The original relation null resampled every pair's replacement target
independently. That quietly destroyed *target reuse*: the first probe set had 27
`IsA` pairs over 9 distinct targets, seven of them ending at `mammal`. Those
seven signatures share the entire `d(mammal, ·)` term by construction, and the
null — with a different random target per pair — shared nothing.

Fed **pure Gaussian random geometry**, containing no semantics whatsoever, the
test then reported `IsA` at p ≤ .05 in 12 of 12 runs, 11 of them surviving
Benjamini-Hochberg. The false-positive rate tracked the target-reuse ratio and
fell to nominal for relations whose targets were all distinct. It was a
target-reuse detector wearing a permutation test's clothes.

Two changes, each verified independently on random geometry:

| null | false positives / 12 runs |
| --- | --- |
| original, independent resampling | 10 |
| multiplicity-preserving resampling | 0 |
| permuted pairing (both multisets exact) | 1 |

and the probe set was rebuilt so every relation has distinct targets, which
restores the power that a correct null would otherwise take away — with reused
targets the correct null converges on the observed data and the test cannot say
anything.

`tests/test_null_calibration.py` runs the analysis on random inputs and fails if
it reports a result. That test is the CI gate.

Result tiers were agreed **before** any real output exists, in
`docs/RESULT_INTERPRETATION_PROTOCOL.md`, so the bar cannot move after the
numbers arrive. `docs/EXPERIMENT_0_FREEZE.md` pins the hashes of the benchmark
and the two core sources, and a test fails if the record drifts from the tree.

### Known limitations, stated plainly

- **There is no non-neural baseline yet.** The geometry null only asks "is there
  *any* shared structure", which two multilingual models trained on overlapping
  web text pass trivially — as would a word-cooccurrence matrix. Until a static
  embedding floor (e.g. fastText en/zh) and a lexical-confound RDM (token
  length, corpus frequency) are added, "independently trained models converge"
  is not yet a falsifiable claim. This is the most important open item.
- **Layer selection is unconstrained.** `select_layers` hill-climbs to maximise
  cross-system agreement and nothing stops it landing on the embedding layer,
  where "geometry" is largely subword overlap. Check `selected_layers` in
  `summary.json` first; if they cluster at 0–2, the result is about tokenisers.
- **The 15 system pairs are not independent.** Three are same-model EN/ZH and
  six are same-language cross-model; only nine speak to cross-model convergence,
  and FDR is currently computed over all 15.
- **Per-relation n is 14–18.** Very little power even with a correct null. Do
  not read a single surviving relation as a result.
- Single bare-word prompt template; no prompt averaging, translation-ambiguity
  or frequency controls, no independent replication model set, no preregistered
  thresholds, and isolated lexical concepts rather than propositions.

### Not evidence

`prototypes/semantic_space_explorer.html` is a synthetic UI mock. Every number
and cluster in it is invented. `archive/` holds a superseded PRH/WIT baseline,
retained as background only.

## What would count as a result

> Held-out cross-model and cross-language semantic geometry, and at least some
> labelled semantic transformations, exceed identity-destroying and
> multiplicity-preserving nulls — and a non-neural baseline — with effects that
> survive multiple-testing correction.

A pretty 2-D projection is not a success criterion.

## Layout

```text
benchmark/   concepts.csv, relations.csv
src/         extraction, analysis, report, ConceptNet overlay
tests/       null calibration, freeze-record drift, notebook sync (this is the CI gate)
scripts/     build_notebook.py -- regenerates the Colab payload from the tree
notebooks/   self-contained Colab run
docs/        EXPERIMENT.md (method + output columns), runbook, references
             DECISION_LOG.md -- why the design is what it is; read before reopening a settled question
             FINDINGS_TO_DATE.md -- what is actually known (currently: one instrument finding, no model results)
             KNOWN_RISKS_AND_OPEN_QUESTIONS.md -- 20-item review checklist
             RESULT_INTERPRETATION_PROTOCOL.md -- result tiers agreed before seeing any real output
             DATA_PROVENANCE.md, EXPERIMENT_0_FREEZE.md -- what the benchmark is, and its pinned hashes
prototypes/  synthetic UI mock (not evidence)
archive/     superseded PRH/WIT baseline
```
