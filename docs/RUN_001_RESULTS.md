# Run 001 — first real pretrained-model results

**Date:** 2026-09-03
**Status:** complete; primary run plus one labelled exploratory sensitivity run
**Verdict against `RESULT_INTERPRETATION_PROTOCOL.md`: Tier 1.**

> Systems share some concept organization, but reusable cross-system semantic
> transformations are not established.

This is the first time any pretrained model has been run in this project.
Everything before it was design.

## What ran

Three model families, two languages, 212 concepts, 145 relation probes, on a
local NVIDIA T600 (4 GB), float32, ~2 minutes of extraction. 1000 permutations,
2000 bootstrap replicates.

The handoff said this needed Colab. It did not: that constraint belonged to the
environment the design was written in, not to the work. A 0.5B model at 212
short prompts is a laptop-GPU job.

## Two defects found before any result was believed

Neither was visible in the code. Both were found by QC on the extracted tensors,
which is now `scripts/qc_representations.py`.

1. **bloom-560m in float16 on CUDA returned NaN** — 80% of the English tensor.
   The extractor saved it without complaint, and `corr()` maps a non-finite
   result to `0.0`, which reads downstream as "these systems do not agree". A
   silent zero, not an error. float32 on the same GPU and inputs is clean
   (max|h| = 1380). The extractor now refuses to save non-finite output, and
   float32 is the default.
2. **`learn` and `study` both carried the Chinese term 学习**, so in every
   Chinese system their representations were byte-identical — a zero distance
   that says nothing about meaning, inside the matrix the whole comparison rests
   on. `study` is now 研究, and a test forbids shared surface forms.

A third, bounded and not fixed: **six Chinese terms (哺乳动物, 鲸, 锤子, 钥匙 and
two others) tokenise to `<unk>` in XGLM only**, 2.8% of the benchmark, making
them mutually indistinguishable in that one system. 鲸 (whale) and 烹饪 (cook)
sit at cosine 1.000000. Qwen and BLOOM cover the set fully.

## The headline number, and why it needs its caveat

Held-out cross-system RDM agreement, against an identity-permutation null:

| | pairs beating null (FDR q<=.05) | mean held-out Spearman rho |
|---|---|---|
| layer chosen freely | **14 / 15** | 0.273 |
| layer restricted to >= 8 | **10 / 15** | 0.190 |

Held-out neighbourhood stability, against a relabelled-system null:

| | observed | null | z | p |
|---|---|---|---|---|
| layer chosen freely | 0.2095 | 0.0661 | 61.7 | 0.001 |
| layer restricted to >= 8 | 0.1402 | 0.0662 | 32.3 | 0.001 |

**The caveat is the layer choice.** Unconstrained selection put five of six
systems at layers 0-3 of 25, and XGLM/en at **layer 0 — the embedding matrix
itself**. This was written into the README as a named risk before the run: *"if
they cluster at 0-2, the result is about tokenisers."* It happened.

So the sensitivity run above is the load-bearing one. Excluding the embedding
and every early layer, the effect shrinks by roughly a third and **does not
vanish**: 10 of 15 pairs still beat the null, and neighbourhood stability still
clears it at z=32. The convergence is not purely lexical.

One correction to how that run should be described. Restricting selection to
layer >= 8 did not land it at mid-depth: it chose **layer 24, the final layer**,
for five of six systems (XGLM/en took layer 12). That follows from the layer
curve, which is U-shaped for BLOOM and Qwen -- agreement is high at layer 1,
collapses through the middle, and recovers at the end.

The full sweep (`scripts/layer_sweep.py`, `layer_sweep.csv`) confirms this and
shows the single cutoff was not a lucky choice:

| min_layer | layers chosen | mean rho | pairs surviving |
|---|---|---|---|
| 0 | 3,3,2,2,0,11 | 0.2734 | 14/15 |
| 2-12 | 24,24,24,24,12,24 | 0.1896 | 10/15 |
| 14-16 | 24,24,24,24,17,24 | 0.1995 | 12/15 |
| 18 | 24,24,24,24,18,24 | 0.2000 | 12/15 |
| 22-24 | all 24 | 0.1866 | 11/15 |

**No mid-depth layer is ever selected.** The moment the earliest layers are
excluded the selector jumps to the last layer and stays there, and the result is
flat at rho 0.187-0.200 across every cutoff from 2 to 24. So the contrast really
is first-versus-last, the middle of these models agrees least, and the reported
"deep" number is not sensitive to the arbitrary choice of 8.

## The finding that was not expected

Restricting to depth splits the pairs cleanly by language, not by model:

| pair | free | deep (>=8) | deep q |
|---|---|---|---|
| bloom·zh vs qwen·zh | 0.432 | **0.448** | 0.003 |
| qwen·zh vs xglm·zh | 0.426 | **0.397** | 0.003 |
| bloom·zh vs xglm·zh | 0.342 | **0.248** | 0.003 |
| qwen·en vs xglm·en | 0.345 | 0.311 | 0.003 |
| bloom·en vs qwen·en | 0.358 | 0.110 | 0.060 (n.s.) |
| bloom·en vs xglm·en | 0.237 | 0.088 | 0.089 (n.s.) |

**Chinese representations agree across models at depth. English ones largely do
not.** Two of three English cross-model pairs lose significance once early
layers are excluded, while all three Chinese pairs hold and the strongest one
gets stronger.

That is the opposite of the naive expectation — these models all saw far more
English. It is a single pilot at 0.5B on 212 bare words and should be treated as
a lead, not a result. Candidate explanations worth testing: English single words
are more polysemous and each model resolves them differently; Chinese words in
this set tokenise more consistently across these three tokenizers; or the
Chinese subset is small enough in training that all three models learned it from
similar sources.

Also worth noting from `cross_language_layer_curve.csv`: BLOOM and Qwen peak in
English/Chinese agreement at **layer 1** (0.252, 0.389), collapse through the
middle, and recover at the last layers. XGLM peaks at **layer 16** (0.273) —
the mid-depth profile you would want to see. The three models do not agree about
where cross-language structure lives.

## The surface-confound control

The identity-permutation null only asks whether *any* shared structure exists.
It cannot tell shared meaning from shared spelling: two tokenizers that split
words alike, and two models that encode string length or token frequency, agree
for reasons with nothing to do with semantics. "14 of 15 pairs beat the null" is
exactly what a purely lexical explanation would also produce.

`src/lexical_controls.py` separates them. For each pair it builds surface RDMs
from properties obtainable without any model — character length, token count,
token-id overlap, and mean token id as a frequency-rank proxy, for *both*
systems' tokenizers — regresses all of it out of both geometries, and correlates
the residuals against a permutation null.

| configuration | pairs surviving (p<=.05) | mean raw rho | mean partial rho |
|---|---|---|---|
| primary (layers 0-3) | **15 / 15** | 0.2734 | 0.2599 |
| last layers (>= 8) | **15 / 15** | 0.1896 | 0.1856 |

Removing surface structure costs about **5%** of the effect in the primary
configuration and **2%** at the last layers. Several pairs go *up*, meaning
surface structure was slightly suppressing their agreement. The controls are not
vacuous — the surface RDMs correlate with the representational geometries at rho
up to 0.46 — they simply are not what the systems agree about.

So the trivial explanation is substantially ruled out for the geometry result.
That is the single most important number in this run.

**What this is not.** It is a confound control, not a competing model. Showing
that these surface features do not explain the agreement is a weaker claim than
showing that a static embedding model like fastText fails to reproduce it. The
non-neural baseline in the next-steps list is still owed, and this does not
substitute for it. The frequency proxy is also a proxy: mean token id is
defensible for frequency-ordered vocabularies but is not a corpus measurement.

### This refines the language asymmetry

The earlier claim that English cross-model pairs "lose significance at depth"
was specific to the identity-permutation null. Under the surface control, at
last layers, they survive: `bloom·en vs qwen·en` at partial rho 0.119 and
`bloom·en vs xglm·en` at 0.117, both p=0.0005. The asymmetry is real but it is
in **magnitude, not existence** — Chinese pairs run 0.25-0.45 where English pairs
run 0.09-0.31. Two tests, two questions; the honest statement is that English
agreement at depth is weak, not absent.

## The static baseline (fastText)

The surface control ruled out spelling. It did not rule out ordinary
distributional semantics, which a 2017 static embedding model also has. That is
the stronger test, and `src/static_baseline.py` runs it: fastText's aligned
vectors become two more systems and go through the same held-out RDM machinery.

Only the 78 held-out concepts fastText covers in both languages are used
(coverage: 84/85 English, 79/85 Chinese).

| comparison | n | mean rho |
|---|---|---|
| language model vs language model | 15 | **0.2847** |
| fastText vs language model | 12 | 0.2190 |
| fastText·en vs fastText·zh | 1 | 0.1861 |

**With the fastText geometry regressed out, LLM-LLM agreement falls only from
0.2847 to 0.2532 — 76-99% retained per pair, and 15 of 15 still significant at
p=0.0005.** So the convergence is not reducible to what a static embedding model
already captures. There is transformer-specific shared structure.

### The part this deflates

Splitting the LLM pairs by language changes the story:

| | mean rho |
|---|---|
| LLM pairs, **same language** (n=6) | **0.3570** |
| LLM pairs, **cross language** (n=9) | 0.2365 |
| fastText·en vs fastText·zh | 0.1861 |

Same-language cross-model convergence sits clearly above everything static. But
**cross-language convergence, at 0.2365, is only about 0.05 above what aligned
fastText vectors achieve between the same two languages.** The cross-language
result — the part closest to "a language-independent semantic space" and the
most interesting-sounding claim available — is the part least distinguishable
from a static aligned baseline. It should be stated that way.

### An internal check that worked

`xglm·en` correlates with `fasttext·en` at **0.383**, higher than it correlates
with most language models. That is exactly what should happen: `xglm·en` is the
system whose selected layer was **0, the embedding matrix**. A system reduced to
its embedding table behaves like a static embedding model. The diagnostic and
the baseline agree about the same defect, which is some evidence both are
measuring what they claim to.

`bloom·en` sits at the other extreme, correlating with `fasttext·en` at only
0.063 — consistent with its effective rank of 18.3, a nearly collapsed space
whose distances carry little structure.

## Relations

Only one relation survives the region-matched null and FDR correction in the
primary run:

| relation | cross-system similarity | effect over region-matched null | q |
|---|---|---|---|
| **IsA** | 0.561 | 0.230 | **0.020** |
| AtLocation | 0.466 | 0.058 | 0.095 |
| HasProperty | 0.770 | 0.016 | 0.113 |
| PartOf | 0.468 | 0.037 | 0.684 |
| everything else | — | <= 0 | ~1.0 |

Every relation now has `target_reuse_ratio` 1.00, so this is not the artefact
that the earlier null produced. Note `HasProperty`: the highest raw similarity
in the table (0.770) and almost no effect over its null. Raw similarity is not
evidence, which is the entire reason the nulls exist.

Within a system, relations do behave like consistent transformations —
`Antonym`, `AtLocation` and `PartOf` clear the strictest permuted-pairing null
in 6 of 6 systems. What does not replicate is the same transformation appearing
in *different* systems.

The relation picture is **stable under the layer re-analysis**, which is the
strongest thing that can be said for it. Exactly one relation survives in both
configurations, and `IsA` gets slightly stronger rather than weaker when the
early layers are excluded:

| relation | effect over null (free) | q | effect over null (last layers) | q |
|---|---|---|---|---|
| **IsA** | 0.230 | **0.020** | **0.251** | **0.020** |
| AtLocation | 0.058 | 0.095 | 0.038 | 0.487 |
| UsedFor | -0.003 | 0.994 | 0.034 | 0.353 |
| PartOf | 0.037 | 0.684 | 0.022 | 0.769 |
| HasProperty | 0.016 | 0.113 | 0.019 | 0.353 |
| everything else | <= 0 | ~1.0 | <= 0 | ~0.86 |

So `IsA` is not an embedding-layer artefact. It is also the relation whose
targets are a coherent set of superordinates (`mammal`, `tool`, `metal`,
`emotion`...), and the region-matched null only partly controls for that. One
relation out of ten, in one pilot, is a lead.

## CORRECTION, 2026-09-04 — the relation section above is withdrawn

Everything above about relations was produced by a statistic that cannot answer
the question it was asked. `relation_convergence.csv` means the per-pair
signatures before comparing them, and

```text
mean_i( d[b_i,.] - d[a_i,.] )  ==  mean(d[targets,.]) - mean(d[sources,.])
```

The pairing cancels exactly. Removing the `sig[a] = sig[b] = nan` self-exclusion
and permuting which source goes with which target reproduces the observed `IsA`
value to eight decimals (0.6835608 over 20 permutations). What little pairing
sensitivity the shipped code showed came from that mask alone. See D17 in
`DECISION_LOG.md`.

Two consequences for the text above. "Only one relation, `IsA`, survives" was
measuring agreement about the direction from the source centroid to the target
centroid, not `IsA` as an operator. And `IsA` cleared the region-matched null
while failing the global one (observed 0.561, global null mean 0.554, p = 0.507,
region-matched null mean 0.331) -- the protocol had asserted region-matched was
the harder of the two without measuring it, and for this statistic it is the
easier.

### The replacement, and two wrong turns on the way to it

Same extraction, same seed, same split, new statistic:
`results/run_001/reanalysis_20260904/`.

The first replacement was matched-versus-mismatched pair signatures under a
*pairing-permutation* null. All ten relations cleared it at p = 0.001 with every
system pair positive, which was the tell: **random non-relational concept pairs
cleared it too**, scoring 0.40-0.48 where real relations scored 0.18-0.33. It was
re-measuring the first-order RDM agreement of `system_alignment.csv`, one pair at
a time. The separation of a pair drives it -- two distant concepts give a large,
well-determined difference vector both systems agree about, while a relation's
source and target sit close together.

The second wrong turn was scoring `matched - mismatched` against a
separation-matched null. That reads a coherent relation as worse than scattered
random pairs, because coherence *raises* the off-diagonal it subtracts. Under it
every relation fell at or below the null, `IsA` at z = -5.2. Both failures are
now asserted by `tests/test_pairing_sensitivity.py`.

The two quantities are reported separately.

**Do systems agree about a specific pair, beyond what its separation implies?**
Null: arbitrary concept pairs held at the same cross-system mean separation.

| relation | matched | null | effect | z | p |
|---|---|---|---|---|---|
| HasProperty | 0.628 | 0.617 | +0.011 | 0.27 | 0.404 |
| Antonym | 0.548 | 0.546 | +0.002 | 0.05 | 0.499 |
| Associated | 0.580 | 0.580 | +0.001 | 0.01 | 0.503 |
| PartOf | 0.537 | 0.580 | -0.043 | -0.99 | 0.840 |
| SimilarTo | 0.498 | 0.563 | -0.065 | -1.49 | 0.929 |
| AtLocation | 0.483 | 0.577 | -0.094 | -2.24 | 0.986 |
| IsA | 0.462 | 0.579 | -0.117 | -3.31 | 0.999 |
| Causes | 0.468 | 0.602 | -0.134 | -3.29 | 0.997 |

**Nothing. No relation exceeds the null, and `IsA`, `Causes` and `AtLocation` sit
significantly below it.** Systems agree *less* about a labelled relation pair than
about two arbitrary concepts the same distance apart.

**Do the relation's own pairs point the same way, across systems?**
Same null. This is the reusable-transformation question.

| relation | coherence | null | effect | z | q |
|---|---|---|---|---|---|
| **HasProperty** | 0.263 | -0.000 | **+0.263** | 21.1 | 0.0014 |
| **UsedFor** | 0.183 | -0.000 | **+0.184** | 16.5 | 0.0014 |
| **PartOf** | 0.083 | 0.000 | +0.082 | 8.2 | 0.0014 |
| **IsA** | 0.072 | 0.000 | +0.072 | 9.1 | 0.0014 |
| **AtLocation** | 0.069 | -0.000 | +0.070 | 7.3 | 0.0014 |
| **HasA** | 0.067 | -0.000 | +0.067 | 6.9 | 0.0014 |
| **Antonym** | 0.039 | 0.000 | +0.039 | 4.5 | 0.0014 |
| SimilarTo | 0.003 | 0.000 | +0.003 | 0.32 | 0.407 |
| Associated | -0.000 | -0.000 | +0.000 | 0.03 | 0.482 |
| Causes | -0.010 | -0.000 | -0.010 | -0.89 | 0.822 |

**Seven of ten, all at q = 0.0014.** This is a larger and more orderly result than
the one it replaces, and it is a different claim: seven relations carry a shared
directional component that recurs across three independently trained models and
two languages, while *no* relation carries pair-specific information that
transfers. The operator exists as a group average and not as something you could
apply to a particular concept -- which is exactly what the retired mean-signature
statistic was measuring all along, correctly, under the wrong name.

The three failures are the three whose members are least alike as a class:
`Causes`, `Associated`, `SimilarTo`. `HasProperty` and `UsedFor` lead by a wide
margin and both have narrow, repetitive target vocabularies, which is a warning
as much as a result.

**The honest caveat on the coherence column.** Its null is arbitrary pairs, whose
mutual similarity is ~0 by construction, so any coherent set of pairs beats it.
It shows the relation is coherent and that the coherence survives crossing
systems; it does not isolate cross-system coherence from within-system coherence
multiplied by the geometry agreement already reported in `system_alignment.csv`.
The next test is that ratio: cross-system coherence against within-system
coherence, per relation. Until it runs, read the coherence column as "there is a
shared direction here", not as "the direction is model-independent".

## CORRECTION, 2026-09-04 — the headline had no denominator

`scripts/prompt_reliability_ceiling.py`. A held-out rho of 0.27 is
uninterpretable without knowing what was achievable: 0.27 of a reachable 0.30 is
near-perfect, 0.27 of a reachable 0.95 is weak, and nothing in this document
distinguished them. Extracting the same model and language twice -- bare terms,
and a mean over six prompt templates with the term's own tokens pooled -- bounds
it.

| system | reliability |
|---|---|
| qwen25_05b·zh | 0.853 |
| qwen25_05b·en | 0.613 |
| bloom_560m·zh | 0.577 |
| **bloom_560m·en** | **0.156** |

**`bloom_560m·en` is not a usable instrument at its selected layer.** Its geometry
barely survives a change of prompt, and it appears in 5 of the 15 system pairs
behind the headline. It also normalises to 1.159 against `qwen25_05b·en` -- it
agrees with another model more than with itself -- which voids that pair rather
than flattering it. This corroborates the effective rank of 18.3 noted above from
an independent direction.

Over the three pairs where both systems clear a 0.30 reliability floor and the
ceiling holds, agreement is **0.468 of what was reachable**: same-language 0.615
(raw 0.432, n=1), cross-language 0.394 (raw 0.264, n=2). The direction matches
the fastText finding -- cross-language is the weak half -- and now has a scale.

XGLM has no prompt-averaged extraction yet, so it is absent here. This is a
stimulus-formulation reliability, not a training-seed ceiling; Pythia publishes
same-architecture seeds and remains the better measurement.

## What this does not show

- Not a Platonic semantic space, not a language-independent internal language.
- The fastText baseline is Wikipedia-trained, and so are these models in part,
  so it is a *non-transformer* baseline rather than an independent one.
- 17 of 212 Chinese terms are absent from fastText (mostly `X的` adjective forms
  and compounds like 计算机, 哺乳动物), so the baseline comparison runs on 78 of
  85 held-out concepts.
- 0.5B models, bare single words, one prompt template, one split seed, one k.
- 2.8% of the Chinese benchmark is `<unk>` in XGLM.
- The mid-depth cutoff of 8 was chosen after seeing the primary layers. It is a
  labelled exploratory sensitivity analysis, not a preregistered choice.

## Next, in order

1. Prompt-averaged and contextualised stimuli instead of bare single words. Both
   remaining weaknesses -- the cross-language result sitting near the static
   baseline, and BLOOM's collapsed English space -- are the kind that bare
   single-word inputs produce.
2. A frequency- and POS-matched null, replacing mean-token-id as a frequency
   proxy with a real corpus measurement.
3. Replication at a larger scale, to test whether the cross-language gap over
   the static baseline widens with model size.
3. Prompt-averaged representations instead of bare words.
4. Fix or exclude the six XGLM `<unk>` concepts.
