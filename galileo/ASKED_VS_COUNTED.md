# Asking a model is not the same as counting its words — and the difference is total

**Result.** On the same ten concepts, what a model *says* about how those concepts
relate has essentially **no relationship** to what you would infer from text the
same model writes. DeepSeek asked, against DeepSeek's own generated corpus:
**ρ = +0.010**.

That is not a weak agreement. It is none.

| | asked directly | inferred from text |
| --- | --- | --- |
| a model agrees with **itself** | **0.974** | 0.750 |
| models agree with **each other** | **0.845** | 0.686 |

Direct elicitation is both far more reliable and shows *more* agreement between
models. Data: `results/elicited_maps.json`, `results/all_corpora.json`.

## The result that inverts the earlier story

Every model's asked map resembles **people's** writing more than it resembles any
model's writing, its own included:

| asked model, compared against… | people's text | its own text |
| --- | --- | --- |
| deepseek-v4-flash | **+0.475** | +0.010 |
| gpt-5.6-luna | **+0.384** | −0.039 (vs deepseek's) |
| glm-5.2 | **+0.366** | +0.110 (vs zhipu's) |

And asked maps are closer to people than the models' own *generated* text was:
counted model-vs-people ran +0.084 to +0.361; asked model-vs-people runs +0.366
to +0.475.

## What this means for the text-derived results

The likeliest explanation is unflattering to the text arm, and it is the same
objection that prompted this work: **the generated corpora largely measure the
prompt, not the model.** Those posts were produced under a topic list, a stance
list and a voice list that this project chose. Their word co-occurrence therefore
reflects that design at least as much as anything about the model. Asking the
model bypasses all of it.

Two consequences, stated plainly:

* **`CONVERGENCE.md` should not be read as a result about models.** Three models
  writing under one prompt design produced similar text; that is now better
  explained by the shared prompt than by shared representation. The direct
  elicitation makes the *same* claim more strongly and without that confound —
  models agree with each other at 0.845 against a 0.974 ceiling — so the claim
  survives, but the evidence for it should be the asked maps, not the counted ones.
* **The text route remains correct for people.** Human posts were not written to
  anyone's prompt. Nothing above impugns `SENTIMENT_DIMENSION.md`, where the
  instrument is applied to writing that already existed.

That asymmetry was in the original design and this project lost it: text
inference for people, who cannot be interviewed; direct questioning for models,
which can.

## Why the elicitation works now

`elicit.py` failed its validity checks in the first pilot and the project
abandoned it. Two of the four remedies its own README listed as untried were
enough — see `README.md`. The one that mattered was recognising that the error
was a *whole-call multiplicative shift* and dividing each call by its own mean,
which cancels it exactly while preserving every ratio inside the call.

| | first pilot | now |
| --- | --- | --- |
| test–retest | 0.827 | **0.929** |
| rod-swap | 0.773 | **0.948** |
| rod ratio CV | 0.151 | **0.077** |

## Limits

* Ten concepts, 45 pairs, one rod (`good`/`evil` = 100).
* Three models, all instruction-tuned, reached through one provider.
* ρ = 0.010 is a *rank* correlation over 45 pairs; it says the two orderings are
  unrelated, not that either is wrong.
* Whether the asked map or the counted map better reflects anything a model
  "really" represents is not settled here. What is settled is that they are not
  interchangeable, and that the choice between them is not a matter of
  convenience.
