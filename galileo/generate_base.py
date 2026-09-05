#!/usr/bin/env python3
"""Arm 3c: a BASE model, to test whether the convergence is assistant register.

Three instruction-tuned models were found to organise politics almost identically
to each other and quite differently from people. The most likely mundane
explanation is not representation at all but **style**: all three are assistants,
post-trained to write in a hedged, balanced, explanatory register, and that shared
register alone could produce a shared semantic structure.

A base model separates those. Qwen2.5-1.5B has had no instruction tuning and no
RLHF; it has never been taught to be an assistant. If convergence with the tuned
models survives, it is not assistant register. If it collapses, the register was
most of the effect. Either answer is worth having, which is what makes this the
right next experiment.

The elicitation has to change, and the change is the point. A base model does not
follow instructions -- asked to "write ten posts about the election" it will
continue the instruction rather than obey it. So it is prompted the way base
models actually work: given a short opener naming a topic and left to continue.

Two constraints carry over unchanged, because the comparison depends on them:
topics are the same ones drawn from the human corpus, and **no human post is ever
shown**. The format scaffold below is written here, not lifted from the corpus.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re

import os

# Triton's JIT cannot build its CUDA shim under WSL here (gcc fails on
# driver.c), and transformers reaches for a compiled path during generate().
# Both are switched off before torch loads; eager attention is what the rest of
# this repository already uses for the same reason.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TRITON_DISABLE", "1")

import torch

# Qwen2's RoPE routes an `aten::bmm` through a fused Triton kernel, and Triton
# cannot build its CUDA shim in this WSL environment -- gcc fails on driver.c for
# want of Python.h, the dev headers not being installed. The router falls back to
# the ordinary aten lowering whenever its condition does not match, and that
# condition consults this predicate by global lookup at call time, so replacing it
# is enough to take the fallback. In float16 the fallback produces non-finite
# logits and a device-side assert in multinomial, so this runs in float32.
from torch._native.ops.bmm_outer_product import triton_impl as _bmm_triton
_bmm_triton._is_outer_product = lambda *a, **k: False

from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent
from generate_posts import TOPICS, STANCES  # same topics as the tuned models

# Written for this script, not taken from the corpus. A bare topic word gives a
# base model no register to continue in and it drifts into encyclopedia prose;
# a minimal feed-shaped opener keeps the continuation post-like without showing
# it anything a person actually wrote.
OPENERS = [
    "Politics feed\n\nOn {topic}: ",
    "What people are saying about {topic}.\n\n\"",
    "Posted about {topic}:\n\n",
    "Someone commented on {topic}. They wrote: \"",
    "Comments on {topic}\n\n- ",
    "Thread: {topic}\n\nReply: ",
    "{topic} \u2014 reactions\n\n",
    "A post about {topic}, written by someone who is {stance}:\n\n",
]

# A 0.5B base model given only a handful of distinct prompts converges on the
# same high-probability continuations: an earlier run produced 482 unique posts
# and then nothing but duplicates for twenty minutes while the GPU stayed busy.
# Widening the prompt space is what makes repeated sampling actually independent.
LEADINS = [
    "", "Honestly, ", "Look, ", "The thing is, ", "Nobody seems to notice that ",
    "After everything, ", "Here is what bothers me: ", "Say what you want, but ",
]


def clean(raw: str) -> list[str]:
    """Every post in a continuation, not just the first.

    A base model given a feed-shaped opener continues with several lines. An
    earlier version kept only the text before the first newline, discarding most
    of every generation and making the run roughly five times longer than it
    needed to be. Each line is its own post, which is also what the opener is
    asking the model to produce.
    """
    out = []
    for line in raw.split("\n"):
        line = re.sub(r"\s+", " ", line).strip(' "\'')
        words = line.split()
        if len(words) >= 8:
            out.append(" ".join(words[:60]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target-tokens", type=int, default=60000,
                    help="content tokens to produce, with margin over the matched budget")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out", default="results/llm_posts_base.jsonl")
    args = ap.parse_args()

    from text_space import tokenize

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager").to(dev).eval()
    actual = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"{args.model} on {dev} as {actual}")

    rng = random.Random(20260905)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")

    seen: set[str] = set()
    total = kept = raw = 0
    stalled = 0
    while total < args.target_tokens:
        before = kept
        prompts, metas = [], []
        for _ in range(args.batch):
            topic, stance = rng.choice(TOPICS), rng.choice(STANCES)
            prompts.append(rng.choice(OPENERS).format(topic=topic, stance=stance)
                           + rng.choice(LEADINS))
            metas.append(f"{topic}|{stance}")
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=1.0, top_p=0.95, top_k=100,
                                 pad_token_id=tok.pad_token_id)
        for i, seq in enumerate(gen):
            for text in clean(tok.decode(seq[enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)):
                raw += 1
                # Base models loop. Exact repeats would inflate PPMI counts for
                # whatever phrase happened to get stuck, so they are dropped --
                # but nothing else is filtered, because filtering on content
                # would shape the very structure being measured.
                if text.lower() in seen:
                    continue
                seen.add(text.lower())
                fh.write(json.dumps({"text": text, "cell": metas[i]}) + "\n")
                kept += 1
                total += len(tokenize(text))
        fh.flush()
        stalled = stalled + 1 if kept == before else 0
        if stalled >= 10:
            print(f"\nstopping early: 10 batches with no new post. The model has "
                  f"exhausted what it will produce for these prompts, at "
                  f"{total:,} content tokens.", flush=True)
            break
        if kept and (kept // args.batch) % 10 == 0:
            print(f"  {kept:,} posts kept of {raw:,}   {total:,}/{args.target_tokens:,} tokens",
                  flush=True)
    fh.close()
    print(f"\n{kept:,} posts kept of {raw:,} generated ({kept/raw:.0%}), "
          f"{total:,} content tokens")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
