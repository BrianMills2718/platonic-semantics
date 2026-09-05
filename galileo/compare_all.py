#!/usr/bin/env python3
"""Do independently built models converge with each other, or with the people they learned from?

This is the question the repository exists for, asked with a text instrument
instead of activations. `compare_arms.py` asked it of one model; this asks it of
several at once, which is what makes the answer interpretable.

One model differing from people proves little on its own -- it could be that
model, that prompt, or that day. But if several models built by different
organisations on different data all differ from people **in the same direction**,
while agreeing with each other more than any of them agrees with people, that is
convergence: a shared structure that is a property of the models rather than of
any one of them, and one that is not simply inherited from the humans they were
trained on.

The decisive comparison is therefore not any single number but two averages:

    mean(model <-> model agreement)   vs   mean(model <-> people agreement)

both read against the ceiling of how well each corpus agrees with itself. If the
first clearly exceeds the second, the models share something people do not.

Every control from the two-corpus version still applies and now binds across all
corpora at once: one shared vocabulary, equal content-token budgets, topics drawn
from the human corpus with no human post ever shown to any model, and a split-half
ceiling per corpus.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import statistics

import numpy as np

from text_space import (load_posts, shared_vocabulary, ppmi_profiles,
                        distance_matrix, spearman, tokenize,
                        spearman_brown)
from compare_arms import content_tokens, take_tokens

ROOT = pathlib.Path(__file__).resolve().parent


def space(posts, vocab, window, max_context):
    return distance_matrix(ppmi_profiles(posts, vocab, window, max_context=max_context))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard",
                    default=str(pathlib.Path.home() / "projects/data/exorde/data_49.parquet"))
    ap.add_argument("--corpus", action="append", default=[], metavar="NAME=PATH",
                    help="a model-written corpus as JSONL; repeatable")
    ap.add_argument("--vocab-size", type=int, default=10)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--max-context", type=int, default=3000)
    ap.add_argument("--match", choices=("tokens", "concepts"), default="tokens",
                    help="equalise total content tokens, or occurrences of the shared "
                         "concepts -- the quantity PPMI actually counts")
    ap.add_argument("--out", default="results/all_corpora.json")
    args = ap.parse_args()

    rng = random.Random(20260904)
    corpora: dict[str, list] = {}
    people = load_posts(args.shard, theme="Politics")
    rng.shuffle(people)
    corpora["people"] = people
    for spec in args.corpus:
        name, _, path = spec.partition("=")
        rows = [json.loads(l) for l in (ROOT / path).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        rng.shuffle(rows)
        corpora[name] = rows

    for k, v in corpora.items():
        t = content_tokens(v)
        print(f"{k:<12s} {len(v):>7,} posts  {t:>9,} content tokens  ({t/len(v):.1f}/post)")

    # Everything is cut to whatever the smallest corpus can supply. PPMI sharpens
    # with text volume, so any corpus allowed more text would look more coherent
    # for free, and that alone would decide the comparison.
    budget = min(content_tokens(v) for v in corpora.values())
    corpora = {k: take_tokens(v, budget) for k, v in corpora.items()}
    print(f"\nall cut to {budget:,} content tokens")

    names = list(corpora)
    vocab = shared_vocabulary(list(corpora.values()), args.vocab_size)
    if len(vocab) < args.vocab_size:
        raise RuntimeError(f"only {len(vocab)} concepts are frequent in every corpus; "
                           "the corpora are not describable over one shared vocabulary")
    print(f"shared concepts ({len(vocab)}): {', '.join(vocab)}")

    vset = set(vocab)

    def occurrences(posts):
        return sum(sum(1 for w in tokenize(p["text"]) if w in vset) for p in posts)

    dens = {k: 1000 * occurrences(v) / max(content_tokens(v), 1) for k, v in corpora.items()}
    print("\nconcept density (shared-vocabulary hits per 1,000 content tokens):")
    for k, v in dens.items():
        print(f"  {k:<10s} {v:>7.1f}")
    print(f"  spread: {max(dens.values())/max(min(dens.values()),1e-9):.2f}x between "
          "the densest and sparsest corpus")

    if args.match == "concepts":
        # Equal tokens is not equal evidence. PPMI is estimated from concept
        # occurrences, so a corpus that mentions the shared concepts more often
        # gets more data for the same token budget, and a density gradient can
        # masquerade as a semantic one.
        def take_occurrences(posts, target):
            out, tot = [], 0
            for p in posts:
                out.append(p)
                tot += sum(1 for w in tokenize(p["text"]) if w in vset)
                if tot >= target:
                    return out
            raise RuntimeError(f"only {tot:,} concept occurrences available, need {target:,}")

        cbudget = min(occurrences(v) for v in corpora.values())
        corpora = {k: take_occurrences(v, cbudget) for k, v in corpora.items()}
        print(f"\nre-cut to {cbudget:,} concept occurrences each "
              f"(tokens now {', '.join(f'{content_tokens(v):,}' for v in corpora.values())})")

    iu = np.triu_indices(len(vocab), 1)
    D = {k: space(v, vocab, args.window, args.max_context) for k, v in corpora.items()}

    ceiling = {}
    for k, v in corpora.items():
        g = list(v); rng.shuffle(g); mid = len(g) // 2
        ceiling[k] = spearman_brown(
            spearman(space(g[:mid], vocab, args.window, args.max_context)[iu],
                     space(g[mid:], vocab, args.window, args.max_context)[iu]))

    print("\nhow well each corpus agrees with ITSELF (the ceiling for any comparison):")
    for k, v in ceiling.items():
        print(f"  {k:<12s} {v:.3f}")

    pair = {}
    for a, b in itertools.combinations(names, 2):
        pair[(a, b)] = spearman(D[a][iu], D[b][iu])

    print("\nagreement between corpora:")
    w = max(len(n) for n in names) + 1
    print(" " * (w + 1) + "".join(f"{n:>{w}s}" for n in names))
    for a in names:
        row = "".join(
            f"{'  --  ':>{w}s}" if a == b
            else f"{pair.get((a, b), pair.get((b, a))):>{w}.3f}" for b in names)
        print(f"{a:<{w}s} {row}")

    models = [n for n in names if n != "people"]
    mm = [v for (a, b), v in pair.items() if a != "people" and b != "people"]
    mp = [v for (a, b), v in pair.items() if a == "people" or b == "people"]

    print(f"\n  model <-> model    mean {statistics.fmean(mm):.3f}   "
          f"({len(mm)} pairs: {', '.join(f'{v:.2f}' for v in mm)})")
    print(f"  model <-> people   mean {statistics.fmean(mp):.3f}   "
          f"({len(mp)} pairs: {', '.join(f'{v:.2f}' for v in mp)})")
    conv = statistics.fmean(mm) - statistics.fmean(mp)
    print(f"  difference         {conv:+.3f}")

    worst = min(ceiling.values())
    if worst < 0.60:
        verdict = ("cannot be read: at least one corpus disagrees with itself too much "
                   "for any comparison to mean anything at this amount of text")
    elif conv > 0.15 and min(mm) > max(mp):
        verdict = (f"CONVERGENCE: all {len(models)} models agree with each other more than "
                   "any of them agrees with people -- and the weakest model-model pair "
                   "still beats the strongest model-people pair, so this is not one "
                   "outlier carrying the average")
    elif conv > 0.15:
        verdict = (f"the {len(models)} models agree with each other more than with people "
                   "on average, but the model-model and model-people ranges overlap, so "
                   "it is not a clean separation")
    elif conv < -0.15:
        verdict = ("the models resemble people more than they resemble each other -- the "
                   "opposite of convergence")
    else:
        verdict = ("no convergence the instrument can resolve: models are about as close "
                   "to people as they are to each other")
    print(f"\n  VERDICT: {verdict}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "vocab": vocab, "window": args.window, "max_context": args.max_context,
        "matched_content_tokens": budget, "match_mode": args.match,
        "n_posts": {k: len(v) for k, v in corpora.items()},
        "ceiling": {k: float(v) for k, v in ceiling.items()},
        "pairwise": {f"{a}|{b}": float(v) for (a, b), v in pair.items()},
        "model_model_mean": statistics.fmean(mm),
        "model_people_mean": statistics.fmean(mp),
        "convergence": conv, "verdict": verdict,
        "distances": {k: v.tolist() for k, v in D.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
