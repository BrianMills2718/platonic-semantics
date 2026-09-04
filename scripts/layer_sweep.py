#!/usr/bin/env python3
"""How does cross-system agreement depend on which layers may be selected?

Run 001 used a single `--min-layer 8` cutoff, and that turned out to resolve to
the final layer for five of six systems, so it tested first-vs-last rather than
depth. This sweeps the cutoff instead and reports the curve, which is the honest
version of that check.

Geometry only: it reuses `analyze_semantic_geometry`'s own layer selection, RDM,
split and permutation machinery, and skips the relation nulls that dominate the
full analysis' runtime. Nothing here re-implements a statistic.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze_semantic_geometry as A  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repr-dir", default="outputs/representations")
    ap.add_argument("--out", default="outputs/analysis/layer_sweep.csv")
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--min-layers", type=int, nargs="*", default=None)
    args = ap.parse_args()

    with (ROOT / "benchmark/concepts.csv").open(encoding="utf-8") as f:
        concept_rows = list(csv.DictReader(f))
    ids = [r["concept_id"] for r in concept_rows]
    selection_idx, eval_idx = A.stratified_split(concept_rows, 0.60, args.seed)

    reps = {}
    for path in sorted((ROOT / args.repr_dir).glob("*__bare.npz")):
        z = np.load(path, allow_pickle=False)
        if [str(x) for x in z["concept_ids"].tolist()] != ids:
            raise SystemExit(f"concept ordering mismatch: {path}")
        reps[f"{z['model_key']}|{z['language']}"] = z["reps"].astype(np.float32)
    if len(reps) < 2:
        raise SystemExit(f"need >=2 representation files in {args.repr_dir}")

    n_layers = min(v.shape[1] for v in reps.values())
    layer_rdms = {s: [A.cosine_rdm(x[:, i, :]) for i in range(x.shape[1])]
                  for s, x in reps.items()}
    cutoffs = args.min_layers if args.min_layers is not None else list(range(0, n_layers, 2))
    print(f"{len(reps)} systems, {n_layers} layers, {len(eval_idx)} held-out concepts")

    rows = []
    for lo in cutoffs:
        if lo >= n_layers:
            continue
        rng = np.random.default_rng(args.seed)
        chosen = A.select_layers(layer_rdms, selection_idx, min_layer=lo)
        selected = {s: layer_rdms[s][chosen[s]] for s in reps}
        rhos, ps = [], []
        for a, b in itertools.combinations(sorted(selected), 2):
            obs, null = A.system_alignment_permutation(
                selected[a], selected[b], eval_idx, rng, args.permutations)
            rhos.append(obs)
            ps.append(A.p_greater(obs, null))
        q = A.bh_fdr(np.array(ps))
        knn = {s: A.knn_sets(A.sub_rdm(d, eval_idx), 10) for s, d in selected.items()}
        rows.append({
            "min_layer": lo,
            "layers_chosen": ",".join(str(chosen[s]) for s in sorted(chosen)),
            "mean_rho": float(np.mean(rhos)),
            "min_rho": float(np.min(rhos)),
            "max_rho": float(np.max(rhos)),
            "pairs_surviving_fdr": int((q <= 0.05).sum()),
            "n_pairs": len(rhos),
            "mean_knn_jaccard": float(np.mean(A.mean_pairwise_jaccard(knn))),
        })
        r = rows[-1]
        print(f"  min_layer {lo:2d}: layers [{r['layers_chosen']}]  "
              f"mean rho {r['mean_rho']:.4f}  survive {r['pairs_surviving_fdr']}/{r['n_pairs']}  "
              f"knn {r['mean_knn_jaccard']:.4f}")

    df = pd.DataFrame(rows)
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
