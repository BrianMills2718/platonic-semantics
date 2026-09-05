#!/usr/bin/env python3
"""Arm 2 vs arm 3: do people and a model organise the same political concepts the same way?

Both corpora go through one instrument (`text_space.py`), so the elicitation is
held constant and the writer is the only thing that varies. That is the property
arm 1 could not deliver: magnitude estimation asks a model a question no
population was ever asked, and the two are then incomparable by construction.

Three controls decide whether any difference found here is real.

* **Token matching, not post matching.** PPMI sharpens with the amount of text it
  saw, so the corpus with more words looks more coherent for free. The model
  writes about half as many words per post as people do, so matching post counts
  would hand it a systematic handicap and manufacture a difference.
* **A shared vocabulary.** Both corpora are described over the same concepts. If
  each picked its own, the two spaces would be incomparable and any number
  reported would be meaningless.
* **A split-half ceiling.** Distances here sit in a narrow band, so a
  cross-corpus correlation cannot be read on its own. Splitting each corpus and
  correlating it against itself gives what sampling noise alone permits. Only a
  cross-corpus value clearly BELOW that ceiling is evidence of a real difference;
  a value at the ceiling means the instrument cannot tell them apart, which is a
  statement about the instrument and not about the writers.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics

import numpy as np

from text_space import (load_posts, shared_vocabulary, ppmi_profiles,
                        distance_matrix, spearman, tokenize,
                        spearman_brown)

ROOT = pathlib.Path(__file__).resolve().parent


def content_tokens(posts):
    return sum(len(tokenize(p["text"])) for p in posts)


def take_tokens(posts, budget):
    """Take posts until the content-token budget is met, so corpora match on text seen."""
    out, total = [], 0
    for p in posts:
        out.append(p)
        total += len(tokenize(p["text"]))
        if total >= budget:
            break
    if total < budget:
        raise RuntimeError(f"only {total:,} content tokens available, needed {budget:,}")
    return out


def space(posts, vocab, window, max_context=3000):
    return distance_matrix(ppmi_profiles(posts, vocab, window, max_context=max_context))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard",
                    default=str(pathlib.Path.home() / "projects/data/exorde/data_49.parquet"),
                    help="durable path -- a session scratchpad is wiped between sessions "
                         "and took this corpus with it once")
    ap.add_argument("--llm-posts", default="results/llm_posts.jsonl")
    ap.add_argument("--vocab-size", type=int, default=10)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--max-context", type=int, default=3000)
    ap.add_argument("--out", default="results/arm2_vs_arm3.json")
    args = ap.parse_args()

    rng = random.Random(20260904)

    human = load_posts(args.shard, theme="Politics")
    rng.shuffle(human)
    llm = [json.loads(l) for l in (ROOT / args.llm_posts).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(llm)

    h_tok, l_tok = content_tokens(human), content_tokens(llm)
    print(f"people : {len(human):>7,} posts  {h_tok:>9,} content tokens"
          f"  ({h_tok/len(human):.1f}/post)")
    print(f"model  : {len(llm):>7,} posts  {l_tok:>9,} content tokens"
          f"  ({l_tok/len(llm):.1f}/post)")

    # The reliability sweep put the working point at 4,000 human posts per half.
    # Both corpora are cut to that many tokens per half so neither sees more text.
    budget = min(h_tok, l_tok) // 2
    human, llm = take_tokens(human, budget * 2), take_tokens(llm, budget * 2)
    print(f"\nboth cut to {budget*2:,} content tokens "
          f"({len(human):,} human posts, {len(llm):,} model posts)")
    print("  matched on TOKENS, not posts: PPMI sharpens with text volume, and the")
    print("  model writes shorter, so equal post counts would favour people for free")

    vocab = shared_vocabulary([human, llm], args.vocab_size)
    print(f"\nshared concepts ({len(vocab)}): {', '.join(vocab)}")

    D = {"people": space(human, vocab, args.window, args.max_context),
         "model": space(llm, vocab, args.window, args.max_context)}
    iu = np.triu_indices(len(vocab), 1)
    agree = spearman(D["people"][iu], D["model"][iu])

    halves, half_D = {}, {}
    for name, grp in (("people", human), ("model", llm)):
        g = list(grp); rng.shuffle(g)
        mid = len(g) // 2
        a, b = space(g[:mid], vocab, args.window, args.max_context), space(g[mid:], vocab, args.window, args.max_context)
        halves[name] = spearman_brown(spearman(a[iu], b[iu]))
        half_D[name] = (a, b)

    ceiling = statistics.fmean(halves.values())
    print("\n  agreement of each corpus WITH ITSELF (split-half):")
    for k, v in halves.items():
        print(f"    {k:7s} rho = {v:.3f}")
    print(f"  ceiling (mean)          {ceiling:.3f}   <- the most any comparison could show")
    print(f"  people vs model         {agree:.3f}")
    gap = ceiling - agree
    print(f"  gap below ceiling       {gap:+.3f}")

    # Deliberately not a pass/fail on one threshold. A hard cut at 0.70 would call
    # 0.69 and 0.71 different answers, which they are not, and inviting a later
    # nudge of that number to reach a conclusion is how a result gets manufactured.
    # What matters is the SIZE of the gap relative to the slack in the ceiling.
    worst = min(halves.values())
    if worst < 0.60:
        verdict = ("cannot be read: each corpus disagrees with itself too much for any "
                   "cross-corpus number to mean anything at this amount of text")
    elif gap > 0.15:
        verdict = ("the model organises these concepts DIFFERENTLY from people -- the gap "
                   "is far larger than the slack in the ceiling")
        if worst < 0.75:
            verdict += (", though the ceiling itself is only moderately reliable here, so "
                        "the size of the difference is less certain than its direction")
    elif gap > 0.05:
        verdict = ("a difference, but not much larger than measurement slack -- more text "
                   "is needed before trusting it")
    else:
        verdict = ("no difference the instrument can resolve -- people and model place "
                   "these concepts alike, to the limit of measurement")
    print(f"\n  VERDICT: {verdict}")

    # Per-concept sampling noise, in the same MDS frame the visual draws.
    from make_visual import mds, procrustes
    a, b = half_D["people"]
    Pa = mds(a)
    Pb = procrustes(Pa, mds(b))
    noise_radius = [float(x) for x in np.linalg.norm(Pa - Pb, axis=1)]

    diff = D["people"] - D["model"]
    print("\nconcept pairs placed most differently:")
    ranked = sorted(((abs(diff[i, j]), vocab[i], vocab[j], D["people"][i, j], D["model"][i, j])
                     for i, j in zip(*iu)), reverse=True)
    for g, a, b, dh, dl in ranked[:8]:
        closer = "people" if dh < dl else "model"
        print(f"  {a:>12s} — {b:<12s}  people {dh:.3f}  model {dl:.3f}"
              f"   ({closer} hold them closer)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "vocab": vocab, "window": args.window,
        "matched_content_tokens": budget * 2,
        "n_posts": {"people": len(human), "model": len(llm)},
        "cross_arm_rho": float(agree),
        "split_half_rho": {k: float(v) for k, v in halves.items()},
        "ceiling": float(ceiling), "gap_below_ceiling": float(gap),
        "verdict": verdict,
        "distances": {k: v.tolist() for k, v in D.items()},
        # How far each concept moves between two halves of the SAME human corpus.
        # Drawn as a circle in the visual: an arrow that stays inside it has not
        # actually gone anywhere, it has only been resampled.
        "noise_radius": noise_radius,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
