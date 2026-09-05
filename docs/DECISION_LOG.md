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

### D18 addendum — the retirement gate, run properly (2026-09-04)

D18 justified removing five documents from a link count. Project Meta's lifecycle
policy is explicit that filename, age, size, similarity and inbound-link counts
identify review *candidates* and never establish semantic eligibility
(`ARCHIVE_POLICY.md`, semantic retirement gate). The gate was run afterwards
rather than before. It passes, and the record it requires is below.

- **Former paths:** `docs/CODING_AGENT_CONTEXT.md`, `docs/CODING_AGENT_PROMPT.md`,
  `docs/HANDOFF_ORIGIN.md`, `docs/HANDOFF_MANIFEST.json`, `docs/PILOT_MANIFEST.json`.
- **Terminal disposition:** `retired_git`, the policy default since the
  2026-09-01 amendment. Git history is the recovery route; no tombstone or
  sidecar file was created, per that ADR.
- **Last containing revision:** `b913eac`. Removed in `2714f19`.
- **Cutoff event:** run 001 on 2026-09-03. Every retired file described the
  project as not yet having run a pretrained model.
- **Promoted claims, verified at `b913eac`:** §3 multi-relational argument and §4
  mathematics rule → `THEORY.md` §3 and §6. §1 origin → `THEORY.md` §0. §5
  primary and relation hypotheses → the README success criterion, which is
  stronger because it also requires a non-neural baseline. §18 geometry,
  neighbourhood, relation and robustness gates → `RESULT_INTERPRETATION_PROTOCOL.md`
  tiers 0-4, which additionally name pooling choices and tokenisation artefacts.
  §2 scientific position → `REFERENCES.md`. §6-§17 duplicated `EXPERIMENT.md`,
  `DATA_PROVENANCE.md` and `EXPERIMENT_0_FREEZE.md` and carried no unique claim.
- **Relationships:** all `lineage_only`. No `blocks_archive`,
  `redirect_before_archive` or `review_required` edge existed; nothing outside the
  five referenced them.
- **Gap the gate caught:** the founding conversation was unreachable from the
  repository. It is now preserved unmodified at
  `docs/origin/2026-09-04_founding_conversation.md` and routed from `THEORY.md` §0.
- **Verification:** 17/17 tests pass; no dangling references; every surviving
  document has inbound links.

## D19 — Run 002 is a scale ladder, and its preprocessing is fixed before the data

Run 001 leaves the project's central question forked (`THEORY.md` §7). Either
there is no shared global structure at this level of description, or 0.5B models
on bare single words sit below the resolution where it appears. The Platonic
Representation Hypothesis predicts the second, since convergence is claimed to
rise with capability. Nothing in run 001 separates them, and almost every other
open item is downstream of which one is true.

Run 002 tests it: three families at three scales, two languages.

| family | small | mid | large |
| --- | --- | --- | --- |
| Qwen2.5 | 0.5B | 1.5B | 3B |
| BLOOM | 560m | 1b7 | 3b |
| XGLM | 564M | 1.7B | 2.9B |

Within-family comparison controls architecture and training data, isolating
scale. Cross-family comparison at each tier is the convergence measurement. The
0.5B tier is run 001, reused unchanged.

### Preprocessing, decided now rather than after

Risks 21 and 22 would otherwise confound the whole comparison: if larger models
have different anisotropy, a scale effect and a preprocessing artefact are
indistinguishable. So the choices are fixed here, in advance.

1. **Representations are mean-centred before the cosine RDM.** On principle, not
   on fit: the dominant common direction in language-model representation space
   is known to carry frequency and length rather than concept identity, and that
   is exactly the confound `lexical_controls.py` targets. Uncentred is retained
   as a labelled sensitivity, never as the headline.
   **Disclosure:** the direction this moves run 001 has already been seen
   (centering raises most pairs; removing the top principal component lowers
   several and raises one). So this is preregistered with respect to run 002's
   *new* data only, and the run-001 recomputation under it is a re-analysis, not
   a prediction.
2. **Anisotropy is reported, not silently inherited.** The mean pairwise cosine
   of every selected layer is recorded beside its result. Any system whose
   selected layer exceeds 0.95 is flagged in the output: its rank ordering is
   being read from under 5% of the cosine range.
3. **The reliability floor is enforced.** Every system needs a prompt-reliability
   of at least 0.30 (`scripts/prompt_reliability_ceiling.py`) to enter the
   headline. Systems below it are reported separately, as `bloom_560m|en` is.
4. **Everything else is unchanged from run 001** — seed, split fraction, k,
   permutation and bootstrap counts, and every null.

### The prediction, stated before the numbers exist

Raw agreement is the weak discriminator: bigger models share more training data,
so cross-family agreement could rise for mundane reasons under either reading.
The lattice account in `THEORY.md` predicts something more specific.

**If reading (b) is right** — models are coarsenings and 0.5B is below the
resolution — then with scale: nesting recovery rises monotonically (currently
5.9% to 22.9% depending on clustering resolution), and per-pair relation transfer
becomes non-zero for at least one relation, where it is currently zero of ten.

**If reading (a) is right**, cross-family agreement may drift up while nesting
stays flat and per-pair transfer stays at zero.

Nesting and per-pair transfer are therefore the primary outcomes. Raw held-out
agreement is secondary and will not settle the fork on its own. A flat nesting
curve across a six-fold scale range is a real negative result for the lattice
account and should be reported as one.

## D20 — Run 003 tests context, because the 7B rung is not reachable here

Run 002 left one half of D19's prediction unmet: per-pair relation transfer stayed
at zero of ten relations across a six-fold scale range, while nesting rose.
`RUN_002_RESULTS.md` lists three ways to settle it. The first is not available.

**Why not 7B.** A 7B model is ~14 GB of weights in bfloat16 against 9 GB of
available RAM and a 4 GB GPU. Measured, not assumed. Four-bit quantisation would
fit, and is rejected: D19 fixed dtype across the ladder precisely so that
precision could not covary with scale, and quantising only the top rung would
reintroduce exactly that confound. A 7B rung needs a different machine and stays
open as the decisive measurement.

**What runs instead.** Option 2, contextualised stimuli. Every result so far
rests on bare single words, so a relation may fail to transfer because its
arguments are ambiguous rather than because relations are not portable. The
prompt-averaging path already exists and is already frozen
(`--average-modes config`, six templates, term-span pooling), so this is a
stimulus change against an unchanged analysis.

It also closes a gap run 002 recorded: no prompt-averaged extraction existed at
the mid and large tiers, so the reliability floor could not be applied there.
This run produces it for all nine models.

### The prediction, before the numbers exist

**If bare single words are the limitation**, contextualised representations
should move per-pair relation transfer: at least one relation reaching q <= 0.05,
or the best p-value falling clearly below run 002's 0.093, at any tier.

**If relations are not portable regardless of stimulus**, transfer stays at zero
of ten with best p in the same band, and the difference between bare and averaged
is confined to reliability.

Per-pair transfer is again the primary outcome. Nesting and raw agreement are
reported for continuity but cannot settle this question, since a stimulus change
moves both for reasons unrelated to relations.

### Fixed in advance

1. Same six templates, same term-span pooling, same L2-normalised averaging as
   the frozen extractor. No new prompt engineering.
2. Same analysis, centred, same seed, split, k, permutations and bootstrap.
3. Bare and averaged are compared within tier and within system. A system whose
   bare-vs-averaged reliability falls below 0.30 is reported separately, as
   `bloom_560m|en` was in run 001.
4. A null result is a result and is reported as one. Two consecutive runs failing
   to move per-pair transfer would make "relations are a property of a set, not
   of a pair" the finding rather than an interim reading.

### D20 addendum — run 003 is CPU-only, and loses one system to it (2026-09-04)

Recorded as a decision taken, not a question. Three environment facts forced it,
all measured rather than assumed:

1. **The GPU cannot run the averaged path.** Qwen's transformers 5.x
   implementation launches a triton JIT kernel, and triton cannot build its CUDA
   helper because the python3.12 development headers are not installed
   (`Python.h: No such file or directory`). Eager attention does not avoid it, and
   `DISABLE_KERNEL_MAPPING`, `USE_TRITON_KERNEL=0` and `DISABLE_KERNELS` all fail.
   The bare path is unaffected, which is why run 002 never hit this.
2. **`bloom_560m` cannot run on CPU.** It returns non-finite representations in
   both bfloat16 and float32 — the device, not the precision. It is finite on GPU,
   which is where run 002's tensors for it came from. It is also the model that
   returned NaN in float16 in run 001, so this is its third numerical failure.
3. Together those are exclusive: the one model that needs the GPU is in a run that
   needs the CPU.

**Decision.** Run 003 runs entirely on CPU, in bfloat16, with eager attention, and
**excludes `bloom_560m`**. Holding device, dtype and attention constant across
every system is what D20's bare-versus-averaged comparison requires; a single
system on a different device would put a device difference inside the comparison
it is meant to control. Run 003's small tier therefore has four systems
(`qwen25_05b`, `xglm_564m`, two languages each) rather than six.

Bare is re-extracted for run 003 rather than reused from run 002, so bare and
averaged differ only in stimulus. Run 002's tensors stay as they are.

**What this costs.** Run 003 cannot speak to BLOOM at 560m, and its small tier is
two families rather than three. Cross-tier comparison within run 003 is therefore
uneven in family composition, and any run-003 number is not directly comparable to
a run-002 number computed under a different attention implementation. The
bare-versus-averaged contrast, which is the preregistered question, is unaffected.

Installing `python3.12-dev` would restore the GPU route and is the fix if this
recurs; it needs root and was not attempted unattended.

## D21 — The Galileo arm becomes an interrogative programme, and is recorded

Everything from arm 1 onward has been executed without an entry here; D20 still
covers run 003. This records the arm and fixes its direction.

**What happened.** Arm 1 (asking models for magnitude estimates) failed its
validity checks in its first pilot, and the project responded by abandoning
elicitation and inferring distances from word co-occurrence instead — for models
as well as for people. That was a mistake twice over. It discarded the
ratio-scale property that lets two respondents be compared without alignment,
and it applied to models an inference method whose justification is that its
subject *cannot be interrogated*.

**What fixed it.** Two of the four remedies the Galileo README had listed as
untried. The error was a whole-call multiplicative shift, so dividing each call
by its own mean cancels it exactly while preserving every within-call ratio;
twenty-five orders rather than five did the rest. Test–retest 0.827 → 0.929,
rod-swap 0.773 → 0.948, rod ratio CV 0.151 → 0.077.

**What it revealed.** Asking a model and counting words in text the same model
wrote are unrelated: ρ = 0.010 on identical concepts. The generated corpora
largely measured the prompt grid this project chose, so `CONVERGENCE.md` should
not be read as evidence about models; the claim survives on the elicited maps,
which carry no such confound.

**Direction.** Elicitation is the primary instrument for models; text inference
is retained for people, who cannot be interviewed; the activations route
(runs 001–003) becomes an independent check rather than the only route. The
programme and its phases are in `GALILEO_PROGRAMME.md`. The critical path is to
draw the elicited space — several models in one frame with no alignment step,
which the ratio scale makes legitimate and which nothing in this repository has
yet produced.

