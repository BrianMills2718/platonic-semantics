# platonic-semantics

Do independently trained language models, in different languages, arrive at the
same *shape* of semantic space — and do labelled semantic relations behave like
reusable transformations inside it?

Run 001 happened on 2026-09-03, and its relation half was re-analysed on
2026-09-04. Short answer: **partly, and not in the way first reported.** Held-out
geometry beat its null in 14 of 15 system pairs, but unconstrained layer
selection had parked five of six systems in the embedding and early layers;
restricting to depth leaves 10 of 15 and cuts the effect by a third.

The original relation result -- "only `IsA` survives" -- is **withdrawn**. It came
from a statistic that is algebraically invariant to which source is paired with
which target, so it could not have tested a transformation. What replaced it says
something better and stranger: **seven of ten relations carry a shared
directional component across models and languages (q = 0.0014), while not one
relation carries pair-specific information that transfers.** The operator exists
as a group average and not as something you could apply to a particular concept.

Read `docs/RUN_001_RESULTS.md` for the full write-up, including a finding that
was not expected: at depth the systems separate **by language, not by model**.

Both falsification checks pass. Regressing out surface lexical structure
(character length, token count, token-id overlap, a frequency proxy) costs ~5% of
the effect, so this is not shared spelling. Regressing out **fastText's** geometry
costs ~11%, with 15/15 pairs still significant, so it is not reducible to static
distributional semantics either.

The honest caveat: same-language cross-model agreement (0.357) sits well above
the static baseline, but **cross-language agreement (0.237) is only ~0.05 above
what aligned fastText vectors already achieve** (0.186). The most exciting-sounding
claim is the least supported one.

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

### A test must be sensitive to the thing it claims to test

Three statistics were tried for the cross-system relation question and two of
them were wrong. Each failure is now asserted by `tests/test_pairing_sensitivity.py`
rather than described.

1. **Mean-signature convergence** (`relation_convergence.csv`, the run-001
   headline). `mean_i(d[b_i,.] - d[a_i,.])` equals
   `mean(d[targets,.]) - mean(d[sources,.])`: the pairing cancels. Twenty pairing
   permutations reproduce the observed `IsA` value to eight decimals. Retained as
   descriptive; it measures agreement about a centroid direction.
2. **Matched-versus-mismatched pair signatures under a pairing-permutation null.**
   All ten relations cleared it at p = 0.001 -- and so did random non-relational
   concept pairs, scoring *higher* than any real relation. It was re-measuring
   first-order geometry agreement one pair at a time. Pair separation drives it.
3. **The same statistic against a separation-matched null**, scored as
   `matched - mismatched`. That subtraction penalises coherence, which is the
   property that makes a relation a relation, so every relation scored below
   scattered random pairs.

What ships reports the two quantities separately against a separation-matched
null: whether systems agree about a *specific* pair (they do not, beyond its
separation), and whether a relation's pairs point the same way across systems
(seven of ten do). See D17 in `docs/DECISION_LOG.md`.

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

- **The headline had no denominator until 2026-09-04.** `scripts/prompt_reliability_ceiling.py`
  measures how much of a system's geometry survives a change of prompt. One of
  the six systems, `bloom_560m·en`, scores 0.156 and is not a usable instrument;
  it appears in 5 of the 15 pairs behind the headline. Over the pairs where both
  systems are reliable, agreement is 0.468 of what was reachable. A training-seed
  ceiling (Pythia publishes same-architecture seeds) is still owed.
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
tests/       null calibration, pairing sensitivity, freeze-record drift, notebook
             sync (this is the CI gate)
scripts/     build_notebook.py -- regenerates the Colab payload from the tree
             prompt_reliability_ceiling.py -- what agreement was achievable at all
notebooks/   self-contained Colab run
docs/        THEORY.md -- what "platonic semantic space" is taken to mean, and
             the predictions that follow; read before the results
             EXPERIMENT.md -- method and output columns; REFERENCES.md -- prior art
             REAL_RUN_RUNBOOK.md -- how to actually run it (local GPU, ~2 min)
             RUN_001_RESULTS.md -- the first run, and the two corrections to it
             RUN_002_RESULTS.md -- the scale ladder, and the half-met prediction
             RUN_003_RESULTS.md -- context, and the first relation that transfers
             DECISION_LOG.md -- why the design is what it is; read before reopening a settled question
             FINDINGS_TO_DATE.md -- what is actually known, with what has been withdrawn
             KNOWN_RISKS_AND_OPEN_QUESTIONS.md -- 26-item review checklist, 6 open from 2026-09-04
             RESULT_INTERPRETATION_PROTOCOL.md -- result tiers agreed before seeing any real output
             DATA_PROVENANCE.md, EXPERIMENT_0_FREEZE.md -- what the benchmark is, and its pinned hashes
results/     run_001/atlas.html -- the readable face of the run, real numbers only
prototypes/  synthetic UI mock (not evidence)
archive/     superseded PRH/WIT baseline
```
