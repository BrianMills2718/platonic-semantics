#!/usr/bin/env python3
"""Build Galileo maps by ASKING models, and compare them to maps inferred from text.

This is the arm the project was supposed to start from. Woelfel's method
interrogates a respondent: every distance is reported as a ratio to one declared
reference pair, which makes two respondents' numbers commensurable with no
alignment step at all. That property is the reason the method exists, and it is
precisely what a co-occurrence count cannot give you.

The text route (`text_space.py`) was built after the first elicitation pilot
failed its validity checks, and it is the right tool for people, who cannot be
interviewed. Applying it to models was a mistake: it infers second-hand what the
model will state directly, and it discards the ratio scale on the way.

`elicit.py` now passes those checks (test-retest 0.929, rod-swap 0.948, rod ratio
CV 0.077), so the direct route is available again. Each model is asked twice,
independently, so its own reliability bounds any comparison -- the same discipline
the text arm uses, applied to the instrument that should have been used first.

The last comparison is the one that answers the objection directly: do the two
methods even agree? If asking and counting produce different spaces for the same
concepts, then which one is used is not a detail of convenience.
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
from scipy.stats import rankdata

from elicit import elicit_averaged, pearson

ROOT = pathlib.Path(__file__).resolve().parent

# The same ten the text arm settled on, so asking and counting can be compared
# on identical concepts rather than on two different studies.
# Taken verbatim from the text arm's shared vocabulary so the asked and
# counted maps describe the SAME concepts. A single mismatched word makes the
# two incomparable, which is how the first run of this ended.
CONCEPTS = ["president", "government", "country", "biden", "war", "state", "power", "party", "vote", "world"]


def to_matrix(concepts, pairs, vals):
    n = len(concepts)
    idx = {c: i for i, c in enumerate(concepts)}
    M = np.zeros((n, n))
    for (a, b), v in zip(pairs, vals):
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = v
    return M


def spearman(a, b):
    # Average ranks for ties. argsort(argsort(x)) hands tied values arbitrary
    # distinct ranks, which biases rho and makes it depend on input order.
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[],
                    help="repeatable; defaults to three families")
    ap.add_argument("--permutations", type=int, default=25)
    ap.add_argument("--budget", type=float, default=2.00)
    ap.add_argument("--text-space", default="results/all_corpora.json",
                    help="text-derived distances to compare against")
    ap.add_argument("--out", default="results/elicited_maps.json")
    args = ap.parse_args()

    models = args.model or ["openrouter/openai/gpt-5.6-luna",
                            "openrouter/z-ai/glm-5.2",
                            "openrouter/deepseek/deepseek-v4-flash"]
    pairs = list(itertools.combinations(CONCEPTS, 2))
    rod = ("good", "evil", 100)
    trace = f"galileo-map-{int(time.time())}"
    iu = np.triu_indices(len(CONCEPTS), 1)

    print(f"{len(CONCEPTS)} concepts, {len(pairs)} pairs, "
          f"{args.permutations} randomised orders per estimate")
    print(f"rod: {rod[0]}/{rod[1]} = {rod[2]}\n")

    D, rel = {}, {}
    for m in models:
        rng = random.Random(20260905)
        short = m.split("/")[-1]
        # Asked twice, independently. Without this the cross-model numbers below
        # have no ceiling and cannot be read.
        e1, _, _ = elicit_averaged(m, pairs, rod, trace, args.budget,
                                   args.permutations, rng, rescale=True)
        e2, _, _ = elicit_averaged(m, pairs, rod, trace, args.budget,
                                   args.permutations, rng, rescale=True)
        rel[short] = pearson(e1, e2)
        mean = [statistics.fmean(p) for p in zip(e1, e2)]
        D[short] = to_matrix(CONCEPTS, pairs, mean)
        print(f"  {short:<22s} agrees with itself  {rel[short]:.3f}")

    names = list(D)
    print("\nagreement between models, ASKED directly:")
    w = max(len(n) for n in names) + 1
    print(" " * (w + 1) + "".join(f"{n[:10]:>12s}" for n in names))
    cross = {}
    for a in names:
        row = ""
        for b in names:
            if a == b:
                row += f"{'  --  ':>12s}"
            else:
                v = spearman(D[a][iu], D[b][iu])
                cross[f"{a}|{b}"] = v
                row += f"{v:>12.3f}"
        print(f"{a:<{w}s} {row}")

    mm = [v for k, v in cross.items() if k.split("|")[0] < k.split("|")[1]]
    ceiling = statistics.fmean(rel.values())
    print(f"\n  model <-> model (asked)   mean {statistics.fmean(mm):.3f}")
    print(f"  ceiling (self-agreement)       {ceiling:.3f}")

    # The comparison that answers the objection to the text route.
    txt = ROOT / args.text_space
    agree_txt = {}
    if txt.exists():
        t = json.loads(txt.read_text(encoding="utf-8"))
        tv = t["vocab"]
        if set(tv) == set(CONCEPTS):
            order = [tv.index(c) for c in CONCEPTS]
            print("\nasked-vs-counted, same concepts, same models:")
            for tname, tmat in t["distances"].items():
                T = np.array(tmat)[np.ix_(order, order)]
                for aname in names:
                    v = spearman(D[aname][iu], T[iu])
                    agree_txt[f"{aname}|text:{tname}"] = v
                    if tname.split("-")[0] in aname or aname.split("-")[0] in tname:
                        print(f"  {aname:<22s} vs its own text corpus   {v:+.3f}")
            best = max(agree_txt.items(), key=lambda kv: kv[1])
            print(f"  best of any asked/counted pairing: {best[0]}  {best[1]:+.3f}")
        else:
            print(f"\n(text space uses different concepts: {sorted(set(tv) ^ set(CONCEPTS))})")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "concepts": CONCEPTS, "rod": list(rod), "permutations": args.permutations,
        "models": names, "self_agreement": rel,
        "cross_model": cross, "ceiling": ceiling,
        "asked_vs_counted": agree_txt,
        "distances": {k: v.tolist() for k, v in D.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
