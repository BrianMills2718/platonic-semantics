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

## Why no other route here can produce this

Activations give a geometry of concepts with no place in it for the model.
Word co-occurrence gives whatever the text happened to mention. The Self-point
costs exactly one more object, and **only asking can obtain it**. This is the
clearest case for the interrogative route in the whole project.

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
