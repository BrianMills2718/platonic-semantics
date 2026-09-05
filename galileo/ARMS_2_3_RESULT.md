# Arms 2 and 3: people vs a model, one instrument

> **Superseded twice.** `CONVERGENCE.md` extends this to three models, and
> `ASKED_VS_COUNTED.md` then shows the whole text route measures this project's
> prompt grid rather than the models. Kept for provenance; do not cite the model
> half as evidence about models.

**Result: a language model asked to post about politics does not organise those
concepts the way people writing about them do.** Agreement between the two is
**0.30**, against a ceiling of **0.84** — where the ceiling is how well each
corpus agrees with *itself* on a different random half of its own text. The gap
is far larger than the slack in that ceiling.

| | |
| --- | --- |
| people | 3,959 posts (Exorde, December 2024, Politics, English) |
| model | 4,877 posts, `gpt-5.6-terra`, topics drawn from the human corpus, no human post ever shown |
| matched at | 59,452 content tokens each |
| concepts | president, biden, trump, government, country, war, state, party, power, world |
| people agree with themselves | 0.828 |
| model agrees with itself | 0.853 |
| people vs model | **0.297** |

Visual: `results/spaces.html`. Data: `results/arm2_vs_arm3.json`.

## Why this comparison is meaningful and arm 1's was not

Arm 1 asked models for magnitude estimates against a standard rod. It failed its
own validity check — the judgments are not stable enough to build a map on, and
no population was ever asked those questions anyway, so there was nothing to
compare against.

Arms 2 and 3 share one instrument (`text_space.py`) that reads distances out of
*usage* rather than asking for them. The same code, the same settings and the
same amount of text are applied to both corpora, so the writer is the only thing
that varies.

## Four controls, each of which could have voided the result

1. **Matched on tokens, not posts.** PPMI sharpens with the volume of text seen.
   The model writes ~12 content tokens per post against the humans' ~15, so equal
   post counts would have handed people sharper estimates for free and
   manufactured a difference. Both corpora are cut to the same token count.
2. **A shared vocabulary.** Both spaces are described over the same ten concepts,
   chosen as frequent in *both* corpora. Letting each pick its own would produce
   two incomparable spaces.
3. **Topics drawn from the human corpus, wording never.** The model was given
   subjects taken from the Exorde shard and never shown a human post. Showing
   examples would have measured imitation instead of representation.
4. **A split-half ceiling.** These distances all sit in a narrow band (0.75–0.91),
   so a cross-corpus correlation means nothing on its own. Splitting each corpus
   and correlating it with itself gives what sampling noise alone permits.

## What the model actually moves

Ordered by how much it rearranges a concept's neighbourhood:

| concept | biggest change |
| --- | --- |
| party | holds it much closer to **trump** than people do |
| country | pushes **president** far away; people keep them close |
| president | pulls **power** in close; pushes **country** away |
| government | pushes **country** and **war** away |
| world | pushes **government** away |

The pattern: people write about politics with `country`, `president` and
`government` tightly bound together — the ordinary vocabulary of a nation's
affairs. The model separates them and instead binds partisan objects
(`party`–`trump`) and abstractions (`president`–`power`) more tightly. It writes
*about* politics as a subject; the human posts are people talking about their own
country.

## Honest limits

* **Superseded in scope.** This two-corpus result stands, but the four-corpus
  version in `CONVERGENCE.md` answers the more interesting question and should be
  read first.
* **One model, one month, one language, one topic.** Nothing here separates
  "language models in general" from "this model", and `Exorde` December 2024 is a
  specific political moment.
* **No ratio scale.** Distances inferred from text have no standard rod, so these
  cannot be compared with arm 1's numbers even if arm 1 had worked. Arms 2 and 3
  are comparable to each other, which is what this comparison needed.

## Two methodological corrections made here

* **Context profiles must be bounded.** PPMI profiles originally ran over every
  context word that ever appeared — tens of thousands, most seen once or twice.
  Those columns carry no signal but do enter the cosine, so measured distance was
  dominated by which rare words happened to fall nearby. Capping the context
  vocabulary at the 3,000 most frequent words is what moved reliability from
  0.65 to 0.71–0.74.
* **Split-half ceilings must be corrected to full length.** A split-half figure
  measures a HALF-sized corpus, while the comparison it bounds uses full corpora,
  so it understates. Uncorrected, this was read as 0.73 rather than 0.84. The
  error only became visible with more corpora, where two different corpora were
  seen agreeing more than a corpus agreed with itself — an impossibility.
  Spearman-Brown is the standard correction.
* **Reliability must be estimated from repeated splits.** A single split-half
  number swings by ±0.15 here. An early sweep on single splits suggested 25,000
  tokens would be plenty; averaged over five splits the true figure was far
  worse, and acting on the single-split number would have produced a confident,
  wrong corpus-size decision.
