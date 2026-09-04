#!/usr/bin/env python3
"""Do the systems disagree like different RESOLUTIONS of one structure, or like different structures?

This is D19's first primary outcome, and it asks something agreement magnitude
cannot. Two models can agree weakly for two very different reasons: they may be
coarse and fine views of one structure, in which case one grouping NESTS inside
the other and clusters only ever merge; or they may have found different
structures, in which case the groupings CROSS.

Measured as conditional entropy. `H(B|A)/H(B)` is the share of B's grouping that
knowing A does not explain; a perfect refinement drives it to zero, crossing
leaves both directions high, and the statistic is the smaller of the two
directions. The null is partitions with the identical cluster-size profile
assigned at random, which removes the mechanical dependence of conditional
entropy on cluster sizes -- without it, fine partitions look nested by
construction.

Always sweep the clustering resolution. On run 001 a single K hid a monotone
trend that was itself the finding. Method recorded as
`lrn-20260904T182317863504Z-a321f37c26`.

    python scripts/nesting_by_scale.py
    python scripts/nesting_by_scale.py --linkage ward --ks 20
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_semantic_geometry import (  # noqa: E402
    cosine_rdm,
    select_layers,
    stratified_split,
)


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def conditional_entropy(b, a):
    """H(B|A): what knowing partition A leaves unexplained about partition B."""
    total = 0.0
    for v in np.unique(a):
        m = a == v
        _, counts = np.unique(b[m], return_counts=True)
        p = counts / counts.sum()
        total += (m.sum() / len(a)) * float(-(p * np.log(p)).sum())
    return total


def unexplained(a, b):
    """Best nesting direction: 0 when either partition perfectly refines the other."""
    return min(
        conditional_entropy(b, a) / max(entropy(b), 1e-9),
        conditional_entropy(a, b) / max(entropy(a), 1e-9),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repr-root", default="outputs/run002",
                    help="directory holding tier_<name>/ subdirectories")
    ap.add_argument("--tiers", nargs="*", default=["small", "mid", "large"])
    ap.add_argument("--ks", nargs="*", type=int, default=[6, 10, 14, 20, 30])
    ap.add_argument("--linkage", default="average", choices=["average", "complete", "ward"])
    ap.add_argument("--null-draws", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--selection-frac", type=float, default=0.60)
    ap.add_argument("--out", default=None, help="optional JSON path")
    args = ap.parse_args()

    with (ROOT / "benchmark/concepts.csv").open(encoding="utf-8") as f:
        concept_rows = list(csv.DictReader(f))
    selection_idx, _ = stratified_split(concept_rows, args.selection_frac, args.seed)
    rng = np.random.default_rng(7)

    print(f"linkage={args.linkage}  null={args.null_draws} draws  seed={args.seed}")
    print(f"{'tier':7s} {'systems':>7s} {'K':>3s} {'observed':>9s} {'null':>7s} "
          f"{'recovered':>10s} {'pairs>null':>11s}")

    results = {}
    for tier in args.tiers:
        d = ROOT / args.repr_root / f"tier_{tier}"
        files = sorted(d.glob("*.npz"))
        if len(files) < 4:
            print(f"{tier:7s} incomplete ({len(files)} files) - skipped")
            continue

        layer_rdms = {}
        for path in files:
            z = np.load(path, allow_pickle=False)
            key = f"{str(z['model_key'])}|{str(z['language'])}"
            reps = z["reps"].astype(np.float32)
            # D19: centred, and layer selection sees only the selection split.
            layer_rdms[key] = [cosine_rdm(reps[:, i, :], center=True)
                               for i in range(reps.shape[1])]
        chosen = select_layers(layer_rdms, selection_idx)

        trees = {}
        for s, layers in layer_rdms.items():
            dd = layers[chosen[s]]
            dd = (dd + dd.T) / 2
            np.fill_diagonal(dd, 0)
            trees[s] = linkage(squareform(dd, checks=False), method=args.linkage)

        per_k = {}
        for k in args.ks:
            parts = {s: fcluster(t, k, criterion="maxclust") for s, t in trees.items()}
            obs, null = [], []
            for a, b in itertools.combinations(sorted(parts), 2):
                A, B = parts[a], parts[b]
                obs.append(unexplained(A, B))
                null.append(np.mean([
                    unexplained(rng.permutation(A), rng.permutation(B))
                    for _ in range(args.null_draws)
                ]))
            obs, null = np.array(obs), np.array(null)
            recovered = (1 - obs.mean() / null.mean()) * 100
            per_k[k] = {
                "observed": round(float(obs.mean()), 4),
                "null": round(float(null.mean()), 4),
                "recovered_pct": round(float(recovered), 1),
                "pairs_beating_null": int((obs < null).sum()),
                "n_pairs": int(len(obs)),
            }
            print(f"{tier:7s} {len(parts):>7d} {k:>3d} {obs.mean():>9.3f} {null.mean():>7.3f} "
                  f"{recovered:>9.1f}% {int((obs < null).sum()):>8}/{len(obs)}")

        results[tier] = {
            "systems": len(layer_rdms),
            "selected_layers": {s: int(chosen[s]) for s in chosen},
            "nesting": per_k,
        }

    if args.out:
        outp = ROOT / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
