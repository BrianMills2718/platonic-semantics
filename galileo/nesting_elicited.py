#!/usr/bin/env python3
"""Do the models disagree like different RESOLUTIONS of one structure, or like different structures?

This is the question `docs/THEORY.md` actually poses, and the end point this line
of work was aimed at. The lattice account says each model is a coarsening of one
shared structure; if so their groupings should **nest** — clusters merge as
resolution drops, never cut across each other. Models that found genuinely
different structures produce crossing groupings instead.

Run 001 tested it on activations over 212 concepts: directionally supported and
weak, 6–23% recovery toward perfect nesting. This asks the same question of
spaces the models were **asked** for rather than read out of their internals — an
independent instrument on the same hypothesis. Agreement between the two would
strengthen the lattice account considerably; disagreement would locate the
problem in one of the instruments.

**Why this needed 40 concepts.** On ten it is degenerate. At K=7 five of seven
clusters are singletons, a mostly-singleton partition nests inside anything, and
the statistic returns a meaningless 100%. Scale here is the difference between a
test and an artifact, which is why `scale_elicit.py` exists.

The statistic and its null are taken unchanged from `scripts/nesting_by_scale.py`
so the elicited and activation results are directly comparable: `H(B|A)/H(B)` is
the share of B's grouping that knowing A does not explain, reported as the
smaller of the two directions, against partitions with the identical cluster-size
profile assigned at random. Without that null, fine partitions look nested by
construction.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib
import random
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from nesting_by_scale import unexplained  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent


def partition(M, k, method="average"):
    return fcluster(linkage(squareform(M, checks=False), method=method),
                    k, criterion="maxclust")


def degenerate(labels):
    """Share of items sitting alone. A mostly-singleton partition nests trivially."""
    sizes = collections.Counter(labels)
    return sum(1 for v in sizes.values() if v == 1) / len(sizes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/scaled_*.json")
    ap.add_argument("--ks", nargs="*", type=int, default=[4, 6, 8, 10, 12, 15])
    ap.add_argument("--nulls", type=int, default=500)
    ap.add_argument("--out", default="results/nesting_elicited.json")
    args = ap.parse_args()

    files = sorted(ROOT.glob(args.glob.replace("results/", "results/")))
    if len(files) < 2:
        raise RuntimeError(f"need at least two spaces, found {len(files)}")

    D, concepts = {}, None
    for f in files:
        j = json.loads(f.read_text())
        name = j["model"].split("/")[-1]
        if concepts is None:
            concepts = j["concepts"]
        elif j["concepts"] != concepts:
            raise RuntimeError(f"{name} used a different concept set; not comparable")
        D[name] = np.array(j["distances"])
    names = list(D)
    print(f"{len(concepts)} concepts, {len(names)} models: {', '.join(names)}\n")

    rng = random.Random(20260905)
    rows = []
    print(f"{'K':>3s} {'singleton share':>16s} {'pairs > chance':>15s} "
          f"{'recovery toward nesting':>24s}")
    for k in args.ks:
        if k >= len(concepts):
            continue
        P = {m: partition(D[m], k) for m in names}
        deg = float(np.mean([degenerate(P[m]) for m in names]))
        recs, wins = [], 0
        for a, b in itertools.combinations(names, 2):
            obs = unexplained(P[a], P[b])
            null = []
            for _ in range(args.nulls):
                x, y = list(P[a]), list(P[b])
                rng.shuffle(x); rng.shuffle(y)
                null.append(unexplained(np.array(x), np.array(y)))
            mu = float(np.mean(null))
            recs.append((mu - obs) / mu if mu > 0 else 0.0)
            wins += obs < mu
        # int() rather than the numpy scalar: `wins` accumulates numpy bools and
        # json refuses int64, which failed the write after the whole test had run.
        rows.append({"k": int(k), "singleton_share": float(deg),
                     "beat_chance": int(wins), "pairs": int(len(recs)),
                     "recovery": float(np.mean(recs))})
        flag = "  <- degenerate" if deg > 0.5 else ""
        print(f"{k:>3d} {deg:>15.0%} {wins:>10d} / {len(recs)} "
              f"{np.mean(recs)*100:>22.1f}%{flag}")

    usable = [r for r in rows if r["singleton_share"] <= 0.5]
    print(f"\n  usable resolutions (under half singletons): "
          f"{[r['k'] for r in usable] or 'NONE'}")
    if usable:
        m = float(np.mean([r["recovery"] for r in usable]))
        w = sum(r["beat_chance"] for r in usable)
        t = sum(r["pairs"] for r in usable)
        print(f"  mean recovery toward nesting: {m*100:.1f}%   "
              f"beating chance {w}/{t}")
        print(f"\n  activations (run 001, 212 concepts): 5.9-22.9%, rising with K")
        print(f"  elicited  (this run, {len(concepts)} concepts): {m*100:.1f}%")
        print("\n  " + ("both instruments say MOSTLY CROSSING with a nesting component"
                        if m < 0.35 else
                        "the elicited spaces nest MORE than the activation spaces did"))
    (ROOT / args.out).write_text(json.dumps(
        {"concepts": concepts, "models": names, "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
