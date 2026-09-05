#!/usr/bin/env python3
"""Elicit far more concepts than fit in one call, by equating batches on common items.

The instrument's reliability comes from judging many pairs together: a whole call
drifts by one multiplicative factor, which is cancelled by dividing the call by
its own mean (`README.md`). That fix is also the ceiling on scale. Ten concepts
is 45 pairs and fits one call; sixty concepts is 1,770 pairs and does not, and
splitting them across calls reintroduces exactly the drift the rescaling removes
— each batch would be normalised to its own mean, so batches would be internally
consistent and mutually incomparable.

This is a solved problem in psychometrics, and the solution is **common-item
equating**. A fixed set of anchor pairs appears in *every* batch. Because the
anchors are judged repeatedly, each batch can be rescaled so its anchors agree
with the reference batch, and every batch is then on one scale. The anchors pay
for themselves: they are also a running check, since a batch whose anchors
disagree wildly with the reference is a batch that should not be stitched in.

Why this matters beyond convenience: the nesting test — whether models disagree
like different *resolutions* of one structure or like different structures — needs
enough concepts to cluster meaningfully. On ten it is degenerate: at K=7 five of
seven clusters are singletons, a mostly-singleton partition nests inside
anything, and the statistic returns a meaningless 100%. Scale is not polish here;
it is the difference between a test and an artifact.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import random
import statistics
import time

import numpy as np

from elicit import elicit, rescale_to_unit_mean

ROOT = pathlib.Path(__file__).resolve().parent


def load_concepts(path, limit):
    rows = list(csv.reader(open(path)))
    words = [r[1].strip().lower() for r in rows[1:] if len(r) > 1 and r[1].strip()]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(w)
    return out[:limit]


def equate(batch_vals, anchor_idx, reference):
    """Put one batch on the reference scale using the anchors it shares with it.

    Ratio-scale judgments differ between batches by a multiplicative constant, so
    the correction is the ratio of anchor means. Returns the scaled values and
    how well the anchors agreed, which is the diagnostic for whether the batch
    should be trusted at all.
    """
    mine = [batch_vals[i] for i in anchor_idx]
    if reference is None:
        return batch_vals, 1.0, float("nan")
    factor = statistics.fmean(reference) / max(statistics.fmean(mine), 1e-9)
    scaled = [v * factor for v in batch_vals]
    after = [scaled[i] for i in anchor_idx]
    # Agreement AFTER equating: a constant is removed, so what remains is whether
    # the anchors keep their relative order and spacing between batches.
    rel = [a / max(b, 1e-9) for a, b in zip(after, reference)]
    return scaled, factor, statistics.pstdev(rel) / max(statistics.fmean(rel), 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openrouter/openai/gpt-5.6-luna")
    ap.add_argument("--concepts", default="../benchmark/concepts.csv")
    ap.add_argument("--n-concepts", type=int, default=40)
    ap.add_argument("--anchors", type=int, default=8,
                    help="pairs repeated in every batch, which put the batches on one scale")
    ap.add_argument("--batch", type=int, default=40, help="new pairs per call")
    ap.add_argument("--permutations", type=int, default=3)
    ap.add_argument("--budget", type=float, default=2.50)
    ap.add_argument("--out", default="results/scaled_space.json")
    args = ap.parse_args()

    concepts = load_concepts(ROOT / args.concepts, args.n_concepts)
    all_pairs = list(itertools.combinations(concepts, 2))
    rng = random.Random(20260905)
    rng.shuffle(all_pairs)

    # Anchors are drawn from the same pool so they are ordinary pairs, not a
    # special stimulus the model might treat differently.
    anchors = all_pairs[:args.anchors]
    rest = all_pairs[args.anchors:]
    batches = [rest[i:i + args.batch] for i in range(0, len(rest), args.batch)]
    rod = ("good", "evil", 100)
    trace = f"galileo-scaled-{int(time.time())}"

    print(f"{len(concepts)} concepts -> {len(all_pairs):,} pairs")
    print(f"{len(batches)} batches of {args.batch}, each carrying the same "
          f"{args.anchors} anchor pairs")
    print(f"{args.permutations} orders per batch -> "
          f"~{len(batches) * args.permutations} calls\n")

    value, reference, drifts = {}, None, []
    for bi, batch in enumerate(batches):
        pairs = anchors + batch
        anchor_idx = list(range(len(anchors)))
        per = []
        for _ in range(args.permutations):
            idx = list(range(len(pairs)))
            rng.shuffle(idx)
            vals, _ = elicit(args.model, [pairs[i] for i in idx], rod, trace, args.budget)
            restored = [0.0] * len(pairs)
            for slot, orig in enumerate(idx):
                restored[orig] = vals[slot]
            per.append(rescale_to_unit_mean(restored))
        mean = [statistics.fmean(c) for c in zip(*per)]
        scaled, factor, drift = equate(mean, anchor_idx, reference)
        if reference is None:
            reference = [scaled[i] for i in anchor_idx]
        else:
            drifts.append(drift)
        for p, v in zip(pairs[len(anchors):], scaled[len(anchors):]):
            value[frozenset(p)] = v
        if bi % 5 == 0 or bi == len(batches) - 1:
            print(f"  batch {bi+1}/{len(batches)}  x{factor:.2f}  "
                  f"anchor spread after equating "
                  f"{'--' if drift != drift else f'{drift:.3f}'}")

    missing = [p for p in all_pairs[args.anchors:] if frozenset(p) not in value]
    if missing:
        raise RuntimeError(f"{len(missing)} pairs never elicited")
    for p, v in zip(anchors, reference):
        value[frozenset(p)] = v

    n = len(concepts)
    idx = {c: i for i, c in enumerate(concepts)}
    M = np.zeros((n, n))
    for p, v in value.items():
        a, b = tuple(p)
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = v

    print(f"\n  batches equated: {len(drifts)}")
    print(f"  anchor spread after equating: median {statistics.median(drifts):.3f}"
          f"  worst {max(drifts):.3f}")
    print("  (0 would mean the anchors land identically once the scale factor is "
          "removed; large values mean a batch should not be trusted)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "concepts": concepts, "rod": list(rod),
        "n_pairs": len(all_pairs), "batches": len(batches),
        "anchors": [list(a) for a in anchors],
        "anchor_drift": drifts, "permutations": args.permutations,
        "distances": M.tolist(),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
