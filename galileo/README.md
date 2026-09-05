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

## Status: the instrument does not currently support the method

Before building any map, `elicit.py --check-stability` tests whether a model's
magnitude judgments are actually ratio-scale. They are not, at least as elicited
here, and the failure reproduces across two unrelated model families.

| condition | test–retest r | rod-swap r | rod ratio CV |
| --- | --- | --- | --- |
| Luna, batched, **same order both runs** | 0.900 | 0.928 | 0.122 |
| Luna, batched, 5 randomised orders | **0.827** | 0.773 | 0.151 |
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

1. **Give the comparison set as context, ask for one pair.** The batch anchors
   and the single question avoids list position — this separates the two effects
   that are currently confounded. It is the obvious experiment and is not yet run.
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

Arm 1 of three. The intended comparison is: models asked directly (here), people
inferred from their own social-media writing, and models writing social-media
posts inferred the same way. Arms 2 and 3 share an instrument, so any difference
between them isolates the entity rather than the elicitation — and neither of
them depends on the magnitude-estimation problem above, since both read distances
out of text rather than asking for them.
