# The interrogative Galileo programme

**Planning path:** `durable_solo` — the work spans sessions and outlives any one
authority window. **Stage:** prototype. **Owner:** Brian, single writer.

## Outcome

> For a researcher mapping semantic space, change *"geometry can only be
> inferred — from model internals, or from statistics of text the model
> wrote — and inference can never be asked why, or conditioned on anything"*
> into *"an instrument that questions a model directly and returns ratio-scale
> spaces, including spaces classic Galileo could not obtain at all"*, within a
> prototype boundary of local analysis plus small metered elicitation.

**Non-goal:** replacing the activations route. Runs 001–003 answer a different
question and remain valid.
**Non-claim:** nothing here establishes which route reflects what a model
"really" represents. It establishes that they differ and that one is
interrogable.

## Why this is worth a programme

Woelfel's every binding constraint is survey logistics, not theory. Pairs grow
as n(n−1)/2 and respondents fatigue, so a study affords **one** question — "how
far apart in meaning?" — asked once per pair, symmetrically, at one moment. A
model respondent removes all four limits at once. The expansion is therefore not
"the same study, bigger", it is *questions the form could never carry*.

## Current truth

| | state |
| --- | --- |
| instrument | **working** — test–retest 0.929, rod-swap 0.948, rod ratio CV 0.077 (`README.md`) |
| asked vs counted | **unrelated**, ρ = 0.010 (`ASKED_VS_COUNTED.md`) |
| three models asked | self-agreement 0.974, mutual 0.845 (`results/elicited_maps.json`) |
| activations route | runs 001–003, 9 models, 212 concepts — complete, and **never compared to elicitation** |
| relation stack | **in flight** |
| decision log | last entry D20 covers run 003; **the whole Galileo arm is unrecorded** |

## Backward map, and the first missing boundary

```
a reviewer sees where models place concepts, and where they disagree
  <- ONE coordinate frame holding several models          <- FIRST MISSING BOUNDARY
  <- ratio-scale distances that need no alignment step        (have this, undrawn)
  <- rescaled magnitude estimates averaged over orders         (working)
  <- a rod and a concept set                                   (working)
```

Everything arm 1 has produced is numbers. **The elicited space has never been
drawn**, and the ratio scale is exactly what makes drawing it legitimate: several
models occupy one frame with no Procrustes step, which is the property the whole
method exists for and which no artifact in this repository yet shows.

## Phases

Only P1 is on the critical path; the outcome is unobserved until it lands.

| | capability | why it could not be done by survey | status |
| --- | --- | --- | --- |
| **P0** | relation-conditioned stack — the same concepts under "power", "harm", "contested", "who benefits" | re-asking every pair per relation multiplies survey length | **done — negative** |
| **P1** | **draw the elicited space: several models, one frame, no alignment** | — | **done** — `results/elicited_map.html` |
| P2 | asymmetry: ask d(A,B) and d(B,A) separately | doubles the survey; classic Galileo assumes symmetry | ready |
| P3 | the self-point: where the model places itself among the objects | a respondent cannot rate themselves on the same rod without priming | ready |
| P4 | motion and the formal inverse: which message moves an object where you want it | needs the same respondent re-surveyed after every candidate message | design required |
| P5 | scale: hundreds of concepts | 100 concepts is 4,950 pairs | conditional on P1 |
| P6 | **bridge the routes on the Self**: one open instruct model, first-person tokens in the concept list, comparing the activation of "yourself" against the elicited Self-point | — | blocked on a model that fits locally |

**Dependency types.** P1 `hard` on the working instrument (satisfied). P2, P3
`optional` — independent, either order. P4 `exploration_required`: "a message"
has no settled meaning for a model respondent, and that is a design question, not
an implementation one. P5 `evidence` on P1 — scaling an unreadable output is
waste.

## Canonical outcome probe

* **Start:** `galileo/results/elicited_maps.json` — three models, ten concepts,
  already elicited.
* **Action:** `python galileo/make_elicited_map.py`
* **Output:** one HTML page, several models in a single frame, with the
  retained-variance figure stated on the page.
* **Step-down:** if two dimensions retain too little, the page says so and shows
  the ordering instead — the rule already applied twice in this repo.
* **Failure case:** a model whose two elicitations disagree must be drawn with an
  error radius that swallows its position, not silently averaged.

Criterion provenance: `explicit_user` — Brian has asked for the spatial
representation repeatedly, and it is the one thing the programme has never
delivered.

## P0 result: meaning did not separate, but the test was weak

Every relation returned the same geometry as the plain question — correlations
0.886 to 0.953 against an instrument ceiling of 0.924, and 0.936 between the
relations themselves. On its face: semantic distance is unitary and the
conditioning buys nothing.

**The concept set makes this result unsafe.** All ten concepts are political, so
"with respect to power", "who benefits" and "how much people argue about them"
are close to synonymous *for these particular concepts*. A null was likely
whatever the model does. The test needs a heterogeneous set — the original pilot's
`dog, wolf, hammer, hospital, anger, justice, river, democracy`, where power and
harm plainly come apart — before the direction is called empty.

## Progress, separated

* **Outcome:** 1 observed — `results/elicited_map.html`, three models in one
  frame from asked distances, rotatable in three dimensions.
* **Enabling:** instrument repaired; asked-vs-counted settled; three models elicited.
* **Process:** this document, and the D21 entry recording the arm.

## Activations vs elicitation: not yet comparable

An earlier draft of this document said the two routes "disagree". **That was
wrong and is retracted.** The ρ = 0.010 result compares elicitation against
*word-counting*, not against activations. Elicitation and activations have never
been compared, and cannot be with what exists:

* **Concepts:** the activations study used 212 concepts; exactly three
  (`government`, `country`, `war`) overlap with the ten elicited. Three concepts
  is three pairs.
* **Models:** activations ran on Qwen, BLOOM and XGLM — open-weight, small,
  base. Elicitation ran on Luna, GLM-5.2 and DeepSeek — API-only, frontier.
  No model appears in both.

The obstacle is structural, not clerical: **activations need open weights,
magnitude estimation needs instruction-following**, and the models that do both
are too large for the local 4 GB GPU. Bridging it needs an open instruct model
that fits, run both ways over one shared concept set — a real experiment, not a
reanalysis, and it belongs in the phase table rather than in a claim.

Until then no route is demoted. Elicitation is the better-measured instrument
(0.974 against 0.750) and the only interrogable one, so it leads the programme;
that is a statement about tractability, not about which better reflects what a
model represents. Nothing measured so far bears on that question.

## Stop / scale / reset triggers

* **Stop** if P1 shows the elicited spaces are too unreliable to place in one
  frame despite the instrument's own checks passing.
* **Scale** to P5 only after P1 is observed.
* **Reset** if a relation-conditioned space turns out to be indistinguishable
  from the classic one across every relation — that would mean the expansion
  direction is empty, and the value moves to P4.
