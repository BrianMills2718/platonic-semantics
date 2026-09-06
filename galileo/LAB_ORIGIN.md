# Does model agreement track the lab that built it? Suggestive, not established

**The question this project could not previously touch.** Every convergence
result here carries one deflationary explanation: models trained on overlapping
corpora may agree because their *sources* overlap, not because they found a
shared structure. Neutral concepts cannot separate those accounts — a platonic
structure and a shared corpus both predict agreement about `dog`. Politically
loaded concepts can.

Three models, 40 concepts elicited in **one run**: 23 loaded (`democracy`,
`censorship`, `capitalism`, `socialism`, `communism`, `protest`, `dissent`,
`revolution`, `freedom`, `authority`, `surveillance`, `propaganda`,
`sovereignty`, `harmony`, `stability`, `patriotism`, `christianity`,
`buddhism`, `atheism`, `tradition`, `individual`, `collective`, `market`) and 17
neutral controls (`sun`, `water`, `tree`, `stone`, `bread`, `iron`, …).

Labs: `glm-5.2` and `deepseek-v4-flash` are Chinese; `gpt-5.6-luna` is American.

## Result

| | same lab (CN–CN) | cross lab | gap |
| --- | --- | --- | --- |
| **loaded** concepts (253 pairs) | +0.670 | +0.583 | **+0.088** |
| **neutral** concepts (136 pairs) | +0.651 | +0.698 | **−0.047** |

**Interaction (loaded gap − neutral gap): +0.135.**
**95% interval, concept-level bootstrap: [−0.038, +0.308]. 92% of draws above zero.**

On neutral concepts the two Chinese models are *not* specially alike — the
cross-lab pairs actually agree slightly more. On loaded concepts they are. The
direction is what training-provenance predicts, and the magnitude clears the
+0.10 threshold registered before the data existed.

**But the interval crosses zero, so this does not establish the claim.**

## The threshold was the wrong instrument, and that is the lesson

`lab_origin.py` pre-registered a threshold on the *point estimate* and printed a
verdict of "support for the training-provenance explanation". Pre-registration
was right; the statistic was not. A threshold with no error bar cannot separate a
real effect from a noisy one, and here the bootstrap immediately showed the
estimate is compatible with zero. The verdict text in that script overstates what
the data carry, and this document supersedes it.

The bootstrap resamples **concepts**, not pairs. Pairs are dependent through
shared concepts, so a pair-level bootstrap would have understated the uncertainty
badly and produced a falsely tight interval.

## What would settle it

* **A second US-lab model.** With one, "cross-lab" and "any comparison involving
  `gpt-5.6-luna`" are the same set, so a quirk of that one model is
  indistinguishable from a US-lab effect. This is the single largest weakness.
* **More loaded concepts.** The interval is wide mostly because 23 concepts is
  few; the bootstrap resamples them and the estimate moves a lot.
* **More labs.** Two CN models from two different labs is a weak basis for
  "same lab"; a third would tell whether the CN–CN agreement is about provenance
  or about those two models.

## What was checked and ruled out

* **Refusal.** A Chinese-lab model declining politically sensitive items would
  make this measure refusal rather than semantics. It did not happen: every
  landed `deepseek-v4-flash` call returned complete, well-formed judgments on
  prompts averaging ~19 loaded concepts, with no empty responses in the
  observability record. Empty arrays in the run log were malformed first attempts
  recovered by retry.
* **Condition asymmetry.** Loaded and neutral concepts were elicited in the same
  run, same rod, same batches, same models, differing only in content.

## Standing

The deflationary explanation is **neither confirmed nor eliminated**. It remains
the strongest alternative to every convergence result in this project, and this
is the first measurement that bears on it at all.
