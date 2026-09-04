# Run 003 — does context make a relation portable?

**Date:** 2026-09-04
**Design:** preregistered in D20 before any tensor was extracted
**Status: small tier complete. Mid and large tiers extracting; replication pending.**
**Verdict so far: the prediction is met, marginally, and in a consistent direction.**

Runs 001 and 002 both found that no relation transfers between models at the level
of a specific pair — zero of ten, at every scale from 0.5B to 3B. Every one of
those results rests on **bare single words**. D20 asked whether that is the
limitation: a relation may only be portable when its arguments are disambiguated
by context.

## What ran

The same 212 concepts and 145 relation probes, extracted twice per system:

- **bare** — the term alone, as in runs 001 and 002;
- **averaged** — the term inside six sentence templates, with only the term's own
  tokens pooled, averaged with each template L2-normalised first.

Everything else is held: same models, same seed, same split, same k, same
permutations, same nulls, same centring, and — deliberately — the same device,
dtype and attention implementation, so bare and averaged differ *only* in
stimulus. Bare was re-extracted rather than reused from run 002 for exactly that
reason.

Run 003 is CPU-only and excludes `bloom_560m`; D20's addendum records why, and
what it costs. The small tier is therefore four systems (Qwen2.5-0.5B and
XGLM-564M, two languages each), giving six system pairs rather than fifteen.

## The preregistered prediction

> **If bare single words are the limitation**, contextualised representations
> should move per-pair relation transfer: at least one relation reaching
> q <= 0.05, or the best p-value falling clearly below run 002's 0.093, at any
> tier.
>
> **If relations are not portable regardless of stimulus**, transfer stays at zero
> of ten with best p in the same band, and the difference between bare and
> averaged is confined to reliability.

## Result: both conditions met, one of them marginally

Per-pair transfer, small tier, bare versus contextualised:

| relation | effect bare | p bare | effect ctx | p ctx |
| --- | --- | --- | --- | --- |
| **HasProperty** | +0.025 | 0.259 | **+0.068** | **0.005** |
| Associated | +0.037 | 0.178 | +0.061 | 0.019 |
| UsedFor | −0.011 | 0.623 | +0.034 | 0.125 |
| Antonym | −0.008 | 0.610 | −0.007 | 0.611 |
| HasA | −0.023 | 0.708 | −0.029 | 0.812 |
| PartOf | −0.027 | 0.756 | −0.031 | 0.821 |
| AtLocation | −0.055 | 0.897 | −0.033 | 0.833 |
| SimilarTo | −0.066 | 0.943 | −0.049 | 0.933 |
| IsA | −0.098 | 0.992 | −0.046 | 0.939 |
| Causes | −0.089 | 0.977 | −0.072 | 0.981 |

| | significant at q ≤ .05 | best p | mean effect |
| --- | --- | --- | --- |
| bare | 0 / 10 | 0.178 | −0.031 |
| **contextualised** | **1 / 10** | **0.005** | −0.010 |

**`HasProperty` is the first relation in this project to transfer between models
at the level of a specific pair.** It is also marginal: q = 0.0500, exactly on the
correction threshold, z = 2.33. Treated alone it is one borderline result out of
ten tests.

The pattern is more persuasive than the single q-value. The three relations with
any positive effect under bare stimuli are the same three that improve under
context, and they improve together — `HasProperty` 0.259 → 0.005, `Associated`
0.178 → 0.019, `UsedFor` 0.623 → 0.125, with `UsedFor` crossing from a negative
effect to a positive one. Mean effect across all ten moves from −0.031 to −0.010.
Nothing that was negative under bare stimuli got meaningfully worse.

Both of D20's conditions are satisfied: a relation reaches q ≤ 0.05, and best p
(0.005) is well below the 0.093 threshold the prediction named.

## What else context changed

| | bare | contextualised |
| --- | --- | --- |
| held-out cross-system agreement | 0.434 | **0.631** |
| pairs beating the null | 6 / 6 | 6 / 6 |
| relations with group-level coherence | 7 / 10 | 7 / 10 |
| mean coherence effect | +0.097 | **+0.155** |
| systems tripping the anisotropy flag | `xglm_564m·zh` | **none** |

Contextualising raises cross-system agreement by about half, strengthens the
group-level coherence that was already there, and clears the near-degenerate
layer problem entirely. That last one matters beyond this run: risk 21 in
`KNOWN_RISKS_AND_OPEN_QUESTIONS.md` has been open since run 001, and a stimulus
change appears to dissolve it.

## The reliability ceiling, and how much of it is used

Both stimulus modes now exist for every included system, so the ceiling is
computable. Reliability is how much of a system's own concept geometry survives
the change of stimulus; agreement between two systems cannot exceed the geometric
mean of theirs except by noise.

| system | reliability |
| --- | --- |
| qwen25_05b·zh | 0.926 |
| xglm_564m·en | 0.922 |
| xglm_564m·zh | 0.870 |
| qwen25_05b·en | 0.718 |

**Every system clears the 0.30 floor, and no pair violates its ceiling** — the
first time that has been true. Run 001 had three of six pairs usable and one
system, `bloom_560m·en`, at 0.156 and unusable outright.

Agreement reaches **0.505 of what was reachable**: same-language 0.601
(raw 0.515), cross-language 0.456 (raw 0.393).

The cross-language figure is the one that moved. Run 001 read same-language 0.615
against cross-language 0.394 — a wide gap, and the basis for saying the
cross-language claim was the weak half. Here the gap is 0.601 against 0.456, much
narrower. Some of that is the corrected pipeline rather than context; run 001's
figures were uncentred and included a broken system. It is a lead, not a result,
and it is not comparable across runs for the reasons in the limits below.

**Correction made while computing this.** `scripts/prompt_reliability_ceiling.py`
was not centring, so it bounded a geometry the pipeline no longer uses and read
about 0.10 low — it reported raw agreement of 0.331 where the analysis it is meant
to bound reports 0.434. It now centres by default, matching D19, with `--no-center`
retained to reproduce pre-D19 numbers. The figures above are the corrected ones.

## How this changes the earlier reading

Run 002 concluded "concepts converge with scale, relations do not." The second
half needs qualifying: relations do not transfer **when concepts are presented as
bare single words**. Given six words of context, at least one does — and the
relations that move are the ones about properties and uses rather than taxonomy.
`IsA`, the relation this project spent the most effort on, remains firmly
negative under both stimuli.

That is a more interesting result than either run alone. It suggests the
bare-word design, not the models, was carrying the negative finding.

## What is still owed

- **Replication at mid and large tiers.** Extracting at time of writing. One tier
  and one borderline q-value is not a finding; if `HasProperty` moves the same way
  at 1.6B and 3B, it is.

  To resume: `scripts/run003_extract.sh` skips any model whose tensors already
  exist, so re-running it continues where it stopped. Then, per tier and mode:

  ```bash
  python src/analyze_semantic_geometry.py \
      --repr-dir outputs/run003/tier_<tier>_<mode> --prompt-mode <mode> \
      --outdir outputs/run003/analysis_<tier>_<mode> --center \
      --permutations 1000 --bootstrap 2000
  ```

  The tier directories are symlink views; build them the same way the small tier's
  were, one per tier and stimulus mode. Extraction is CPU-bound and roughly six
  times slower for the averaged mode, which is the whole cost of this run.
- ~~The prompt-reliability ceiling for run 003~~ — **done, small tier.** See below.
- A 7B rung, still out of reach on this hardware (D20).

## Limits specific to this run

- Four systems and six pairs at the small tier, against fifteen pairs in run 002.
  Less power, and `bloom_560m`'s exclusion means two families rather than three.
- q = 0.0500 is exactly on the threshold. A single additional test in the family,
  or a slightly different correction, would move it across.
- Run 003 numbers are not directly comparable to run 002 numbers: attention
  implementation differs (eager throughout here), and the system set differs. The
  bare-versus-contextualised contrast is internally matched and is the only
  comparison this run licenses.
- The six templates are the frozen set from Experiment 0.2; no prompt engineering
  was done for this run, and none should be without a new freeze.
