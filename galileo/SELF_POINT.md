# The Self: three models place themselves outside politics, and disagree about where

**The Self is Woelfel's most consequential object.** It is not special machinery
— it is judged against the same rod as everything else (*"how far apart are you
and war?"*). That is what makes Galileo predictive rather than merely
descriptive: the distance between the Self and an object forecasts behaviour
toward it, and attitude change *is* that distance shrinking. Woelfel found the
brand nearer a person's Self held the larger market share.

Figure: `results/self_point.html`. Data: `results/self_point.json`.

## Result 1 — the Self is outside the space, categorically

Ten political concepts sit a median **0.85** apart and never more than **1.30**.
The nearest any model places itself to any of them is **1.36**.

**All ten Self-distances exceed the largest distance between any two concepts**,
and it holds within each model separately, not only pooled:

| model | concepts to each other | its own Self to them |
| --- | --- | --- |
| gpt-5.6-luna | mean 0.87, max 1.25 | mean 1.59, **min 1.52** |
| glm-5.2 | mean 0.89, max 1.44 | mean 1.51, **min 1.31** |
| deepseek-v4-flash | mean 0.91, max 1.44 | mean 1.39, **min 1.20** |

These models do not place themselves at the edge of politics. They place
themselves *outside* it — a categorical result, not a matter of degree.

## Result 2 — they agree about politics, not about themselves

| | agreement |
| --- | --- |
| about the ten concepts | **+0.872** |
| about where they stand | **+0.632** |

The same three models that agree strongly about how political concepts relate
agree markedly less about their own position relative to them. The disagreement
is specifically about the Self.

Ordering within that outside position is still legible: nearest is **vote**, then
**state** and **party**; furthest is **war**, then **world** and **biden**. In
Woelfel's framing that ordering is the part that would predict behaviour.

## Can the activations route reach the Self? — corrected

An earlier version of this document said activations "give a geometry of
concepts with no place in it for the model". **That is wrong.** The extraction
pipeline takes an arbitrary concept list (`--concepts`), so `yourself`, `me` or
`I` can be pushed through and given an activation exactly like any other word.
Nothing prevents it, and the existing 212-concept set already carries `machine`,
`computer` and `person`.

The real distinction is narrower and worth stating properly. Feeding "yourself"
through a model yields its representation of **the English word** — the concept
of self-reference, which every model learned from broadly similar text and which
should therefore look broadly similar across models. The elicited Self is the
model answering **as an agent** about its own position. Whether those are the
same thing is an open question, not a settled one.

**One piece of evidence that they may differ**, from data already in hand. If the
Self were simply a shared word, three models should agree about it roughly as
well as they agree about other shared words. Per-object cross-model agreement:

| object | agreement | | object | agreement |
| --- | --- | --- | --- | --- |
| government | +0.947 | | world | +0.855 |
| president | +0.943 | | vote | +0.851 |
| party | +0.911 | | country | +0.851 |
| biden | +0.907 | | war | +0.657 |
| state | +0.887 | | **yourself** | **+0.632** |
| power | +0.859 | | | |

The Self is the single most disputed object. **The margin over `war` is only
0.025**, so this is suggestive rather than decisive — but it is the direction
predicted if the Self indexes something model-specific rather than a shared
lexical meaning.

## The experiment this opens

Put first-person tokens into the activations concept list and compare the
activation-derived position of "yourself" against the elicited Self-point, in one
model measured both ways. If they coincide, the elicited Self is the word and
nothing more. If they diverge, the two routes are measuring different things and
the interrogative one reaches something activations do not.

This is a sharper version of the route-bridging phase (P6 in
`../docs/GALILEO_PROGRAMME.md`) than comparing 212 generic concepts, because the
Self is exactly where the two routes would most plausibly come apart. It carries
the same blocker: an open instruct model small enough to run locally, since
activations need weights and magnitude estimation needs instruction-following.

## Limits, stated plainly

* **This does not establish that a model has a self-representation.** "Yourself"
  may evoke a persona rather than report a structure. What the numbers support is
  narrower: the reports are reproducible within a model (self-agreement 0.95–0.97)
  and comparable between models because of the shared rod.
* **The reliability figure covers the whole matrix**, not the Self row
  specifically. A Self-row reliability would need the two estimates stored
  separately, which this run did not do. Result 2 rests on a correlation over ten
  values and should be treated as indicative.
* **Distance-from-Self predicts behaviour in Woelfel's human studies.** Nothing
  here tests whether it predicts anything about a model. That is the validation
  this result most needs and does not have.
* One rod, ten political concepts, three instruction-tuned models via one
  provider.
