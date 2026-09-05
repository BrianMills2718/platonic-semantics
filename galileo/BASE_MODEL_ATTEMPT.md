# The base-model test: attempted, and it cannot be run here

**Outcome: negative, and the reason is structural rather than fixable by effort.**
A base model was meant to test whether the convergence among three
instruction-tuned models is really shared *assistant register* rather than shared
semantic structure. It could not supply a usable corpus, and even if it had, two
confounds would have made the number uninterpretable.

## What blocked it

**Qwen2.5-0.5B exhausts its own output distribution at ~17,000 content tokens.**
Across two runs with progressively wider prompt spaces (8 openers × 16 topics ×
6 stances × 8 lead-ins, temperature 1.0, top-k 100) it produced 1,332 unique
posts and then nothing but exact duplicates — zero new posts per minute while the
GPU sat at 49% utilisation. This is not slowness; more time yields nothing.

The other corpora were matched at 51,698 content tokens. The base model reached
33% of that.

**Forcing the comparison down to what it can supply destroys the instrument for
every corpus, not just for it.** A matched-corpus design cuts everything to the
smallest member, so a weak arm does not merely add a weak arm:

| corpus | self-agreement at 51,698 tokens | at 17,097 tokens |
| --- | --- | --- |
| people | 0.74 | **0.39** |
| openai | 0.75 | **0.58** |
| deepseek | 0.75 | **0.31** |
| zhipu | 0.72 | **0.58** |

At the base model's budget nothing self-agrees well enough to be compared to
anything, and the guard in `compare_all.py` correctly refuses to report a verdict.
Raw numbers are in `results/with_base.json` and should not be read as a result.

## Why it would not have settled the question anyway

Both of these were stated before the numbers were seen.

1. **Capacity is confounded with tuning.** The comparison models are frontier
   scale; the only local base checkpoints are 0.5B–3B. A divergence could be the
   absence of instruction tuning, or simply the absence of capacity. Nothing in
   the design separates them.
2. **Genre is confounded with tuning.** A base model cannot be asked to write in
   a genre — it continues text rather than following instructions. Its output here
   is encyclopedic prose, not posts. It could be shown examples, but the only
   examples available are human posts (barred by the no-human-posts control) or
   tuned-model posts (which would contaminate the very comparison being made).

## What a clean test needs

A **base checkpoint at a scale comparable to the tuned models**. Every model on
the `llm_client` allowlist is instruction-tuned, so this is not available through
the governed API route, and local hardware (a 4 GB T600) will not hold one. This
is a hardware and model-access problem, not an analysis problem.

## What was learned anyway

The register objection is *partly* answered without a base model, by measuring
register directly rather than assuming it. `genre_check.py` shows the three tuned
models differ substantially in surface register — OpenAI asks questions a third
as often as DeepSeek, and uses numerals a fifth as often as Zhipu — while still
agreeing semantically at 0.66 and 0.64. Shared register is therefore not a
sufficient explanation for the convergence. See `CONVERGENCE.md`.

## Three environment defects fixed on the way

Recorded because they will recur on this machine, and each cost real time.

* **Triton cannot build its CUDA shim under WSL here.** `gcc` fails on
  `driver.c` for want of `Python.h`; the Python dev headers are not installed.
  Qwen2's RoPE routes an `aten::bmm` through a fused Triton kernel, so
  `generate()` dies. Torch's native-op router falls back to the ordinary aten
  lowering when its condition does not match, and that condition consults
  `_is_outer_product` by global lookup at call time — replacing it takes the
  fallback without touching site-packages.
* **That fallback produces non-finite logits in float16**, surfacing as a
  device-side assert inside `torch.multinomial`. It runs in float32.
* **The generator discarded everything after the first newline** of each sample,
  throwing away most of every generation and making the run roughly five times
  longer than necessary. A base model given a feed-shaped opener continues with
  several lines and each is a post.
