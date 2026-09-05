# Three models converge with each other, and not with us

**Result.** Three language models built by three different organisations — OpenAI,
DeepSeek and Zhipu — organise ten political concepts almost identically to one
another, and quite differently from the people whose writing they were trained
on.

|  | mean agreement |
| --- | --- |
| model ↔ model | **0.686** |
| model ↔ people | **0.207** |
| ceiling (a corpus vs itself) | 0.72 – 0.75 |

The models agree with each other about as well as a corpus agrees with itself —
that is, as well as this instrument can measure anything at all. They agree with
people barely at all.

|  | people | openai | deepseek | zhipu |
| --- | --- | --- | --- | --- |
| **people** | *0.74* | 0.36 | 0.08 | 0.17 |
| **openai** | 0.36 | *0.75* | 0.66 | 0.64 |
| **deepseek** | 0.08 | 0.66 | *0.75* | 0.76 |
| **zhipu** | 0.17 | 0.64 | 0.76 | *0.72* |

The separation is clean, not an average hiding an outlier: the **weakest**
model–model pair (0.64) still beats the **strongest** model–people pair (0.36).

Visual: `results/spaces.html`. Data: `results/all_corpora.json`.
Run: `compare_all.py --corpus name=path ...`

## Why this is the interesting version of the question

One model differing from people proves little — it could be that model, that
prompt, or that day. Convergence needs several models built independently. If
they all differ from people *in the same direction*, and resemble each other more
than any resembles people, then the shared structure is a property of models
rather than of any one of them — and, importantly, is not simply inherited from
the humans they learned from.

That is the platonic-representation claim, asked here of **text the models
produce** rather than of their activations. It needs no access to weights, and
the same instrument reads human and model corpora, so nothing has to be aligned
across incommensurable spaces.

## Controls

Each of these could have produced the result on its own if omitted.

1. **Equal text.** All four corpora cut to 51,698 content tokens. PPMI sharpens
   with volume, so a corpus allowed more text looks more coherent for free.
2. **One shared vocabulary.** All four described over the same ten concepts,
   chosen as frequent in *every* corpus.
3. **No human post shown to any model.** Models got topics drawn from the human
   corpus and nothing else. Showing examples would measure imitation.
4. **A per-corpus reliability ceiling**, split-half and Spearman–Brown corrected,
   so every agreement number is read against what sampling noise alone permits.
5. **One identical instrument** for all four corpora — same code, same settings.
6. **Matched on concept occurrences as well as on tokens.** PPMI is estimated from
   occurrences of the shared concepts, not from tokens, and these corpora differ in
   how densely they use them — 34.8 hits per 1,000 content tokens for people
   against 53.1 for DeepSeek, a 1.53× spread. Matching tokens alone therefore
   hands the denser corpora more evidence, and a density gradient could
   masquerade as a semantic one. Re-matching on equal concept occurrences
   (`--match concepts`) leaves the result essentially unchanged:

   | | model ↔ model | model ↔ people | gap |
   | --- | --- | --- | --- |
   | matched on tokens | 0.686 | 0.207 | +0.480 |
   | matched on concept occurrences | 0.669 | 0.190 | +0.479 |

   The weakest model–model pair (0.56) still beats the strongest model–people
   pair (0.31). Data: `results/all_corpora_concepts.json`.

   This control was added after it nearly overturned a different result in this
   project — see `SENTIMENT_DIMENSION.md`, where the same density gradient tracked
   the effect being measured and had to be removed before the finding could stand.

## What would break this

* **Shared training data.** The most likely mundane explanation: three models
  trained on overlapping web corpora may converge because their *sources* overlap,
  not because of anything deeper. This experiment cannot separate those, and that
  is its single largest weakness.
* **Shared post-training style.** All three are instruction-tuned assistants
  asked to write posts, so some of the convergence could be assistant register
  rather than semantic structure.

  **Partly answered, and the answer favours the result.** `genre_check.py`
  measures surface register directly, and the three models are *not* alike on it:

  | | mean words | first person | questions | numbers |
  | --- | --- | --- | --- | --- |
  | people | 31.9 | 38% | 16.9% | 22.5% |
  | openai | 25.7 | 37% | 5.8% | 2.7% |
  | deepseek | 24.5 | 55% | 16.0% | 12.6% |
  | zhipu | 31.0 | 57% | 10.7% | 18.6% |

  OpenAI asks questions a third as often as DeepSeek and uses numerals a fifth as
  often as Zhipu. Yet those two pairs agree semantically at 0.66 and 0.64. If a
  shared register were producing the convergence, models that visibly differ in
  register should not cluster this tightly — so register is not a sufficient
  explanation.

  *An earlier version of this table sampled the first 6,000 human posts in file
  order rather than at random, which inflated their mean length from 31.9 to 37.7
  words and made people look like a length outlier. Parquet row order is not
  arbitrary. The sampling is fixed and the numbers above are the corrected ones;
  the argument is unchanged, because it rests on differences among the models
  rather than on the human row.* It is not fully excluded either: these are surface features, and
  a deeper shared "assistant-ness" would not show up in them.

  The clean test is still a base model, which is why one was attempted (below).
* **The prompt is common to all three.** They received identical topics, stances
  and voices. That is required for comparability but does impose shared
  structure; how much is unmeasured.

Those are the three experiments this result most needs, in that order. The
second is partly addressed above; the base-model attempt is documented in
`BASE_MODEL_ATTEMPT.md`.

## Honest limits

* One month of one platform, English, politics only.
* Ten concepts. The comparison is over 45 pairs.
* Distances are not ratio-scale, so these cannot be compared to arm 1's
  magnitude estimates — but arms 2 and 3 are directly comparable to each other,
  which is what the question needed.
* Two of the three models are Chinese-lab (DeepSeek, Zhipu) and those two agree
  most closely (0.76). A wider spread of labs would test whether the block is
  really "models" or partly "models trained on similar corpora".

## Cost

$0.79 for both new corpora (DeepSeek $0.039, Zhipu $0.752), plus the $1.88
already spent on the OpenAI corpus.
