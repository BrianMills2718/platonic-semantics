# The semantic space recovers the political sentiment scale

**Result.** Five groups of people, separated only by how positive or negative
their political writing is, were each measured over the same ten concepts. The
instrument never sees sentiment — it sees only which words occur near which. The
space it recovers nevertheless lines the five groups up in sentiment order.

Of the **120** possible orderings of five groups, the true sentiment order gives
the strongest relationship between a group's position and its semantic distance
from the others: **ρ = 0.680**, which is the highest value *any* ordering
achieves. Only it and its mirror reach it — **p = 0.017**, exact enumeration.

(Corrected 2026-09-27. This was first reported as ρ = 0.794, and the
concept-matched figure below as 0.721. Band distances are tied by construction
-- four pairs sit one band apart -- and the Spearman used then gave tied values
arbitrary distinct ranks instead of their average rank, inflating ρ. With ties
ranked correctly both values fall; the ordering is still the unique best and p
is unchanged, so the conclusion stands. `tests/test_audit_fixes.py` locks this.)

Figure: `results/sentiment_dimension.html`. Data: `results/human_only_space.json`.

## Why this is the Woelfel-shaped result

Galileo's payload was never one map of concepts. It was the comparison *between*
respondents — Bush voters and Dukakis voters each carrying a space — and the
recovery of an attitude as a direction in that space. This is that: an attitude
dimension (sentiment) recovered as a geometric one, from usage alone, without the
attitude ever being shown to the instrument.

## What is and is not established

* **The ordering is real.** Tested globally by exact permutation, not eyeballed
  from a monotone-looking staircase — which is a shape five noisy points produce
  often enough to be worth guarding against.
* **Adjacent groups are not separable.** Only 1 of 10 pairs (most hostile vs
  warmest, 0.372) exceeds the combined sampling noise of its two members. The two
  middle bands sit 0.016 apart with error bars of 0.26 and 0.39 and swap order on
  the first axis. Nothing should be read into which of them comes first; the
  claim is about the sequence as a whole.
* **Two confounds were tested, and neither explains it.**

  *Different topics.* If hostile and warm groups simply wrote about different
  things, the geometry would track topic rather than sentiment. They do not: the
  largest difference in topic mix between any two of the five groups is **0.079**
  on a 0–1 scale, i.e. they discuss the same concepts in nearly the same
  proportions.

  *Different concept density.* This one was real and had been missed. Every
  concept is used more often by the more hostile groups, monotonically — `trump`
  appears in 14.6% of the most hostile group's posts against 10.3% of the
  warmest's. PPMI is estimated from concept *occurrences*, not from tokens, so
  matching token counts hands the denser groups more evidence and a pure density
  gradient could masquerade as a semantic one. Re-matching all five groups on
  equal occurrences of the shared concepts (`--match concepts`) weakens the
  effect but does not remove it: **ρ = 0.636, still the best of all 120 orderings,
  p = 0.017**. Data: `results/human_matched_concepts.json`.

* **Causation is still open.** Sentiment could organise the semantics, or some
  third property of these writers could drive both. Nothing here settles that.

## Why the numbers here beat the combined map

These groups agree with themselves at **0.83–0.89**. In `respondent_map.html`,
where the same groups sit alongside three models, they agree with themselves at
only **0.47–0.71** and their differences are not resolvable.

The reason is the matched-corpus rule: every corpus is cut to the smallest, and
the smallest was a model corpus at 51,698 content tokens. Removing the models
lets each human group have **103,273**, and the measurement roughly doubles in
precision. A weak arm does not merely add a weak arm — it degrades every other
one, so comparisons that do not need it should not include it.
