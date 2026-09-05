# Meaning separates — but only on concepts that let it

**Result.** Asked under different relations, a model returns genuinely different
spaces over the same concepts — but only one relation actually separates, and the
first test of this was rigged by its own concept set.

Data: `results/relation_stack_hetero.json` (heterogeneous),
`results/relation_stack.json` (political). Run:
`relation_stack.py --concepts heterogeneous`.

## The first run was uninformative, and the concept set is why

On ten **political** concepts — `president, government, country, biden, war,
state, power, party, vote, world` — every relation returned the same geometry as
the plain "how far apart in meaning" question (0.886–0.953 against a 0.924
ceiling). That looked like a clean negative: meaning is unitary, conditioning
buys nothing.

It was an artifact. Asking "how far apart **with respect to power**" about
`government`, `party` and `power` is barely a different question, because those
concepts are already about power. The relations were near-synonymous *for that
set*, so a null was likely whatever the model does.

## On heterogeneous concepts it separates

Rerun on `dog, wolf, hammer, hospital, anger, justice, river, democracy`, where
the relations can come apart:

| | agreement with the classic question | its own reliability | |
| --- | --- | --- | --- |
| who benefits from each one | +0.935 | 0.889 | indistinguishable |
| how much people argue about them | +0.935 | 0.819 | indistinguishable |
| how much power each one holds | +0.894 | 0.780 | indistinguishable |
| **whether each one is good or harmful** | **+0.612** | 0.834 | **a different geometry** |

Relations agree with each other at **+0.733** against a ceiling of **+0.854**, so
the stack is not one space measured four times.

**The separable component is evaluation.** "Good or harmful" is not only distinct
from the plain question (0.612 against its own 0.834 ceiling) — it also disagrees
with every other relation (0.481–0.640). It is its own axis. Power, contestedness
and who-benefits all collapse onto ordinary meaning.

That resonates with Osgood's semantic differential, where evaluation (good–bad)
is the dominant factor and potency (strong–weak) a weaker separate one. This is
not a replication — different method, eight concepts, one model — but the
direction is the same, and potency failing to separate here where evaluation does
is the more interesting half.

## What this costs

Conditioning is less reliable than not conditioning. The classic question
reproduces itself at 0.947; the conditioned ones at 0.780–0.889. Asking a model
to hold a relation in mind while estimating magnitudes is a harder task, and the
stack should be read with that in mind rather than treated as four equally solid
spaces.

## Limits

* One model (`gpt-5.6-luna`), eight concepts, 28 pairs, 12 orders per estimate.
* Four relations, chosen by hand to be plausibly independent. A different four
  could give a different answer, and nothing here says these are the axes.
* The political null is explained, not refuted — on concepts that are all about
  power, these relations genuinely may not differ.

## Why this matters for the programme

This is the first capability in `../docs/GALILEO_PROGRAMME.md` that a human
survey could not have delivered: re-asking every pair under four framings
multiplies the instrument by four, which is fine for a model and impossible for a
fatigued respondent. It also came within one run of being recorded as a negative
result, which would have closed the direction on evidence that could not have
shown anything else.
