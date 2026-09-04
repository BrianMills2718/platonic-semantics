#!/usr/bin/env python3
"""Does cross-system agreement survive removing surface lexical structure?

The identity-permutation null in `analyze_semantic_geometry.py` only asks whether
*any* shared structure exists between two systems. It cannot distinguish shared
meaning from shared spelling: two tokenizers that split words similarly, and two
models that encode string length or token frequency, will agree for reasons that
have nothing to do with semantics. Run 001 reported 14 of 15 pairs beating that
null, which is exactly the result a purely lexical explanation would also
produce.

This is the check that separates them. For each system pair it builds surface
RDMs from properties available without any model -- character length, token
count, token-id overlap, and mean token id as a frequency-rank proxy, for both
systems' tokenizers -- then computes the partial Spearman correlation between
the two representational geometries with all of that regressed out.

If the partial correlation collapses toward zero, the convergence was lexical.
If it survives, something beyond surface form is shared. Reported alongside a
permutation null on the residuals so "survives" is a test, not an eyeball.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]


def cosine_rdm(x):
    x = np.asarray(x, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


def tri(d, idx):
    sub = d[np.ix_(idx, idx)]
    return sub[np.triu_indices(len(sub), 1)]


def surface_features(terms, tokenizer):
    """Per-concept surface properties, computed without running the model."""
    ids = [tokenizer(t)["input_ids"] for t in terms]
    return {
        "chars": np.array([len(t) for t in terms], dtype=float),
        "ntok": np.array([len(i) for i in ids], dtype=float),
        "meanid": np.array([float(np.mean(i)) if i else 0.0 for i in ids], dtype=float),
        "idsets": [set(i) for i in ids],
    }


def surface_rdms(feat, idx):
    """Pairwise surface dissimilarities over the held-out concepts."""
    out = {}
    for key in ("chars", "ntok", "meanid"):
        v = feat[key][idx]
        out[key] = np.abs(v[:, None] - v[None, :])[np.triu_indices(len(idx), 1)]
    sets = [feat["idsets"][i] for i in idx]
    n = len(sets)
    jac = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            u = len(sets[a] | sets[b])
            jac[a, b] = jac[b, a] = 1.0 - (len(sets[a] & sets[b]) / u if u else 0.0)
    out["idjaccard"] = jac[np.triu_indices(n, 1)]
    return out


def rank01(v):
    r = rankdata(v, method="average")
    return (r - r.mean()) / max(r.std(), 1e-12)


def residualise(y, X):
    """Remove everything the surface design matrix can linearly explain."""
    Xd = np.column_stack([np.ones(len(y))] + [rank01(c) for c in X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta


def partial_spearman(a, b, controls, rng, n_perm):
    ra, rb = rank01(a), rank01(b)
    ea, eb = residualise(ra, controls), residualise(rb, controls)
    obs = float(np.corrcoef(ea, eb)[0, 1])
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.corrcoef(ea, rng.permutation(eb))[0, 1]
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    z = float((obs - null.mean()) / max(null.std(ddof=1), 1e-12))
    return obs, p, z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repr-dir", default="outputs/representations")
    ap.add_argument("--analysis-dir", default="outputs/analysis",
                    help="run whose selected layers and concept split are reused")
    ap.add_argument("--out", default="outputs/analysis/lexical_control.csv")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rng = np.random.default_rng(args.seed)
    with (ROOT / "benchmark/concepts.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["concept_id"] for r in rows]

    summary = json.loads((ROOT / args.analysis_dir / "summary.json").read_text())
    chosen = summary["selected_layers"]
    split = pd.read_csv(ROOT / args.analysis_dir / "concept_split.csv")
    eval_idx = np.array([i for i, r in enumerate(split.itertuples())
                         if r.split == "evaluation"], dtype=int)
    print(f"reusing {args.analysis_dir}: {len(eval_idx)} held-out concepts, "
          f"layers {chosen}")

    cfg = json.loads((ROOT / "experiment_config.json").read_text())
    tokenizers = {k: AutoTokenizer.from_pretrained(v) for k, v in cfg["models"].items()}

    selected, feats = {}, {}
    for path in sorted((ROOT / args.repr_dir).glob("*__bare.npz")):
        z = np.load(path, allow_pickle=False)
        key = f"{z['model_key']}|{z['language']}"
        if key not in chosen:
            continue
        selected[key] = cosine_rdm(z["reps"].astype(np.float32)[:, chosen[key], :])
        model_key, lang = str(z["model_key"]), str(z["language"])
        feats[key] = surface_rdms(
            surface_features([r[lang] for r in rows], tokenizers[model_key]), eval_idx)

    out = []
    for a, b in itertools.combinations(sorted(selected), 2):
        va, vb = tri(selected[a], eval_idx), tri(selected[b], eval_idx)
        raw = float(spearmanr(va, vb).statistic)
        controls = [feats[a][k] for k in sorted(feats[a])] + \
                   [feats[b][k] for k in sorted(feats[b])]
        part, p, z = partial_spearman(va, vb, controls, rng, args.permutations)
        # A constant surface RDM (e.g. every term one token) has no defined
        # correlation; skip it rather than letting NaN win the max.
        def _max_surface(v, f):
            vals = [abs(float(spearmanr(v, c).statistic)) for c in f.values()
                    if np.ptp(c) > 0]
            return max(vals) if vals else float("nan")

        surf_a, surf_b = _max_surface(va, feats[a]), _max_surface(vb, feats[b])
        out.append({
            "system_a": a, "system_b": b,
            "raw_spearman": raw,
            "partial_spearman": part,
            "retained_fraction": part / raw if raw else np.nan,
            "permutation_p": p, "z_vs_null": z,
            "max_surface_rho_a": surf_a, "max_surface_rho_b": surf_b,
        })

    df = pd.DataFrame(out).sort_values("partial_spearman", ascending=False)
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"\n{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")
    print(f"\nwrote {dest}")
    print(f"pairs surviving surface control at p<=.05: "
          f"{int((df.permutation_p <= 0.05).sum())}/{len(df)}   "
          f"mean raw {df.raw_spearman.mean():.4f} -> partial {df.partial_spearman.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
