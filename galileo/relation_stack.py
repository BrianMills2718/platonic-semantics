#!/usr/bin/env python3
"""Galileo beyond the survey: one concept set, many relations, a stack of spaces.

Every binding constraint in Woelfel's method comes from the cost of a human
survey, not from the theory. Pairs grow as n(n-1)/2, respondents fatigue, so a
study affords **one** question -- "how far apart are these in meaning?" -- asked
once per pair. Meaning is then measured as a single undifferentiated quantity
because nothing else fits in the instrument.

A model respondent has no fatigue, so the question can be asked repeatedly under
different framings at trivial cost. That yields a *stack* of spaces over the same
concepts: how far apart with respect to power, to morality, to time, to who
benefits. Each is a full Galileo space in its own right, all in the same units
because they share a rod.

The stack answers something the classic method cannot pose at all:

* **Is "meaning" separable?** If every relation returns the same geometry, the
  unconditioned space is all there is and the relations are a distinction without
  a difference. If they diverge, semantic distance has components, and the
  unconditioned question was averaging over them.
* **Which relation is the unconditioned question actually measuring?** Correlate
  the plain "in meaning" space against each conditioned one. Whichever it tracks
  is what respondents have been answering all along.

Two guards, because a stack of spaces makes it easy to see structure that is not
there:

* Every relation is elicited **twice**, so its own reliability bounds any claim
  about how it differs from another. A relation that cannot reproduce itself
  cannot be said to differ from anything.
* Judgments are rescaled per call, the fix that made this instrument work at all
  (see `README.md`); without it a whole-call drift is read as a relation effect.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import statistics
import time

import numpy as np

from elicit import elicit_averaged, pearson
from elicit import PILOT_CONCEPTS
from elicit_map import CONCEPTS, to_matrix, spearman

ROOT = pathlib.Path(__file__).resolve().parent

# `None` is the classic Galileo question, kept as the reference the others are
# compared against. The rest are chosen to be plausibly independent of one
# another rather than paraphrases, which would guarantee a null result.
RELATIONS = [
    None,
    "how much power each one holds",
    "whether each one is good or harmful",
    "how much people argue about them",
    "who benefits from each one",
]


def label(r):
    return "in meaning (classic)" if r is None else r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openrouter/openai/gpt-5.6-luna")
    ap.add_argument("--permutations", type=int, default=12)
    ap.add_argument("--budget", type=float, default=2.00)
    ap.add_argument("--concepts", choices=("political", "heterogeneous"),
                    default="political",
                    help="the political set makes the relations near-synonymous -- "
                         "'power' and 'who benefits' barely differ for `government` "
                         "and `party` -- so a null result there is uninformative. The "
                         "heterogeneous pilot set is where they can come apart.")
    ap.add_argument("--out", default="results/relation_stack.json")
    args = ap.parse_args()

    concepts = CONCEPTS if args.concepts == "political" else PILOT_CONCEPTS
    pairs = list(itertools.combinations(concepts, 2))
    rod = ("good", "evil", 100)
    trace = f"galileo-relations-{int(time.time())}"
    iu = np.triu_indices(len(concepts), 1)

    print(f"{args.model}")
    print(f"{args.concepts} set: {len(concepts)} concepts, {len(pairs)} pairs, "
          f"{len(RELATIONS)} relations, "
          f"{args.permutations} orders x 2 estimates each\n")

    D, rel = {}, {}
    for r in RELATIONS:
        rng = random.Random(20260905)
        e1, _, _ = elicit_averaged(args.model, pairs, rod, trace, args.budget,
                                   args.permutations, rng, relation=r, rescale=True)
        e2, _, _ = elicit_averaged(args.model, pairs, rod, trace, args.budget,
                                   args.permutations, rng, relation=r, rescale=True)
        name = label(r)
        rel[name] = pearson(e1, e2)
        D[name] = to_matrix(concepts, pairs, [statistics.fmean(p) for p in zip(e1, e2)])
        print(f"  {name:<34s} agrees with itself  {rel[name]:.3f}")

    names = list(D)
    ceiling = statistics.fmean(rel.values())
    print(f"\nagreement between the relation-spaces (ceiling {ceiling:.3f}):")
    w = max(len(n) for n in names) + 1
    cross = {}
    for a in names:
        row = ""
        for b in names:
            if a == b:
                row += f"{'  --  ':>9s}"
            else:
                v = spearman(D[a][iu], D[b][iu])
                cross[f"{a}|{b}"] = v
                row += f"{v:>9.3f}"
        print(f"{a:<{w}s}{row}")

    classic = label(None)
    print(f"\nwhat is the classic question actually measuring?")
    tracks = sorted(((cross[f"{classic}|{n}"], n) for n in names if n != classic),
                    reverse=True)
    for v, n in tracks:
        # A conditioned space can only be said to differ from the classic one by
        # more than the instrument's own slack.
        head = min(rel[classic], rel[n])
        verdict = ("indistinguishable from it" if v >= head - 0.05
                   else "clearly a different geometry")
        print(f"  vs {n:<34s} {v:+.3f}   (own ceiling {head:.3f}) -- {verdict}")

    off = [v for k, v in cross.items()
           if k.split("|")[0] < k.split("|")[1] and classic not in k]
    print(f"\n  relations vs each other, mean {statistics.fmean(off):+.3f}")
    print(f"  ceiling                       {ceiling:+.3f}")
    if statistics.fmean(off) >= ceiling - 0.05:
        print("  -> meaning does NOT separate: every framing returns one geometry")
    else:
        print("  -> meaning SEPARATES: asking under different relations gives "
              "genuinely different spaces")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "concept_set": args.concepts, "concepts": concepts,
        "rod": list(rod),
        "relations": [label(r) for r in RELATIONS],
        "permutations": args.permutations,
        "self_agreement": rel, "ceiling": ceiling, "cross_relation": cross,
        "distances": {k: v.tolist() for k, v in D.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
