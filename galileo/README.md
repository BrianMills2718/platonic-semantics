# Galileo — Woelfel's method, with models as respondents

Joseph Woelfel's Galileo (Woelfel & Fink, 1980) measures a semantic space by
asking respondents for **magnitude estimates against a standard rod**: *"good and
evil are 100 units apart — how far apart are dog and wolf?"* Those judgments are
ratio-scale, so two respondents' numbers are in the same units with no alignment
step, and metric MDS turns them into coordinates. Cognitive process is then
defined as *motion* of objects in that space, and the method inverts: you can
solve for which associations would move an object where you want it.

The reason this is worth trying here: **cosine distances from two different
models are not commensurable** — that incomparability is why the rest of this
repository needs RDM correlation, Procrustes alignment and permutation nulls just
to compare two systems. Ratio judgments against a shared rod would be directly
comparable. It would also supply the two things the activation-based route
cannot: a **self-point** (ask the model where *it* sits among the objects) and an
**actionable inverse**.

## Status: the instrument works, after two fixes

Before building any map, `elicit.py` tests whether a model's magnitude judgments
are actually ratio-scale. **They now pass**, after applying two of the four
remedies this file had listed as untried:

| | first pilot | after the fix | wanted |
| --- | --- | --- | --- |
| test–retest r | 0.827 | **0.929** | > 0.9 |
| rod-swap r | 0.773 | **0.948** | > 0.9 |
| rod ratio CV | 0.151 | **0.077** | < 0.2 |

The two changes, in order of how much they mattered:

**1. Rescale every call by its own mean.** The diagnosis below was that the error
is *list-level* — a whole call shifts together — which is why averaging
permutations made things worse rather than better. If that shift is
multiplicative, dividing each call by its own mean removes it exactly while
leaving untouched every ratio inside the call, and ratios are the only thing the
method claims to measure. Checked against the stored pilot data before being
used: order sensitivity falls from 0.475 to 0.195 for GLM-5.2 and 0.236 to 0.189
for Luna.

**2. More permutations.** Five was never enough to average a per-call effect once
that effect was removable. Twenty-five carries test–retest from 0.890 to 0.929.

The rod-ratio test is computed on **raw** magnitudes, not rescaled ones. Rescaling
normalises absolute size away, so a ratio test run on rescaled values reads 1.0
by construction and proves nothing.

### Why this matters more than it looks

The whole point of Galileo is that respondents are *asked*. An earlier detour
inferred distances from word co-occurrence in text instead, which is exactly the
second-hand inference the method exists to replace, and it quietly abandoned the
ratio scale that makes two respondents' numbers comparable without alignment.
That route is still useful for people, who cannot be interviewed — but for models
there is no reason to settle for it, and this instrument is the reason.

| condition | test–retest r | rod-swap r | rod ratio CV |
| --- | --- | --- | --- |
| Luna, batched, **same order both runs** | 0.900 | 0.928 | 0.122 |
| Luna, batched, 5 randomised orders | 0.827 | 0.773 | 0.151 |
| **Luna, 25 orders, per-call rescaling** | **0.929** | **0.948** | **0.077** |
| Luna, one pair per call **+ concept set shown** | 0.511 | 0.525 | 0.407 |
| Luna, **one pair per call** | 0.417 | 0.211 | 1.702 |
| GLM-5.2, batched, 5 randomised orders | 0.758 | 0.786 | 0.171 |

Wanted: r > 0.9 and ratio CV < 0.2. Eight concepts, 28 pairs, rods
`good/evil = 100` and `hot/cold = 50`.

### Three things this found, in the order they were learned

**1. The first reliability number was inflated by its own design.** The naive
test–retest of 0.900 ran both repetitions with the pairs in the same order, so
whatever bias list position introduced was reproduced identically in both runs.
That measures *reproducibility of a bias*, not reliability. Randomising the order
drops it to 0.827, which is the honest figure.

**2. Averaging over presentation orders made it worse, which is the diagnosis.**
If the error were independent per-pair noise, averaging five permutations would
have improved test–retest. It fell. So the error is **list-level**: a whole call
shifts together, every judgment inside it shares the displacement, and reordering
cannot average away something that moves as a block. Order sensitivity — the
spread of one pair's answer across presentation orders — is CV 0.236 for Luna and
0.475 for GLM-5.2.

**3. Removing the list destroys the measurement entirely.** One pair per call was
meant to eliminate list effects at the source. Test–retest collapsed to 0.417 and
the rod ratio's CV blew out to 1.702, with answers ranging 25 to 680 for the same
eight concepts. **The list was doing the anchoring, not the rod.** The model
calibrates by comparing pairs against each other within a batch; given a single
pair and a reference, it has no frame and effectively guesses a magnitude.

That last point is the substantive finding, and it is a real difference from human
respondents. Woelfel's respondents carry a stable internal scale between
questions. These models do not — they reconstruct one from whatever comparison
set is in front of them.

**4. Showing the concept set does not restore the anchor.** The obvious remedy —
give the model the full concept range as context, but ask for one pair, so it can
calibrate without a list of answers to sit inside — recovers only a little:
test–retest 0.417 → 0.511, still far below the batched 0.827.

That completes a clean monotone ordering across four designs, and it is the
result:

> **Stability tracks how many pairs are judged *in the same pass*, not how much
> the model knows about the range.** Batched (0.827) > one pair with the concept
> set (0.511) > one pair alone (0.417).

Knowing which concepts are in play is worth about a fifth of what actually
judging them together is worth. The model is not calibrating against a remembered
scale at all — it is calibrating against the other judgments it is making at that
moment. Woelfel's respondents carry a scale between questions; these models
construct one per call and discard it.

### The one encouraging number

Rod-swap ratio CV sits at 0.12–0.17 in every batched condition. Swapping
`good/evil = 100` for `hot/cold = 50` rescales the whole distance set by a
roughly constant factor, which is genuine ratio-scale behaviour. And the constant
is not the naive 0.50 — it is 0.66–0.88, meaning the model treats "hot to cold"
as a genuinely shorter *semantic* distance than "good to evil" rather than doing
the arithmetic on the declared numbers. A respondent honouring the rod's meaning
would do exactly that.

So the ratio property is partly present. It is the *anchoring* that fails.

## What would be tried next

In rough order of expected value:

1. ~~Give the comparison set as context, ask for one pair.~~ **Run. Recovers
   little — see finding 4 above.**
2. **Anchor with worked examples** rather than one rod: several pre-set distances
   spanning the range, so the scale is pinned at more than one point.
3. **More permutations.** Five was not enough to average a block effect, but the
   trend across 5 → 20 would say whether it converges or plateaus.
4. **A reasoning model.** `reasoning_effort` is deliberately `none` here, because
   magnitude estimation is meant to be an immediate judgment rather than a
   deliberation. That choice is defensible but untested — it may be what costs
   the stability.

## Files

| Path | What |
| --- | --- |
| `prompts/magnitude_estimation.yaml` | The elicitation, as a prompt asset. Deliberately asks for unbounded distances, never a 0–1 similarity, which would discard the ratio property the method exists to obtain. |
| `elicit.py` | Elicitation plus the four validity checks. Exits non-zero when the judgments fail them. |
| `results/stability*.json` | Raw runs for every condition in the table above. |

Calls go through the governed `llm_client` — prompt as data, `json_schema`
output, required `task`/`trace_id`/`max_budget`. The whole pilot cost **$0.0094
across 103 calls**.

## Where this sits

**Arm 1 is now the primary instrument for models**, which reverses what this
section previously said.

The original plan was three arms: models asked directly (here), people inferred
from their own social-media writing, and models writing social-media posts
inferred the same way. When arm 1 failed its first validity check the project
moved to text inference for models too — and that was wrong twice over. It
discarded the ratio scale that lets two respondents be compared without
alignment, and it applied to models an inference method whose whole justification
is that its subject *cannot be interrogated*.

Asking and counting turn out to be unrelated: ρ = 0.010 on identical concepts
(`ASKED_VS_COUNTED.md`). Direct elicitation is also far the better instrument —
self-agreement 0.974 against 0.750. So arm 1 leads, text inference is retained
for people, who cannot be interviewed, and the programme is in
`../docs/GALILEO_PROGRAMME.md`.
