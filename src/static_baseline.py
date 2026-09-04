#!/usr/bin/env python3
"""Do the language models agree with each other beyond what fastText already explains?

The surface-confound control showed the cross-system agreement is not spelling.
It did not show the agreement is anything more than ordinary distributional
semantics, which a 2017 static embedding model also has. This is that test.

fastText's aligned vectors become two more systems (`fasttext|en`, `fasttext|zh`)
and the same held-out RDM machinery applies, giving three comparisons:

1. how well a static model reproduces each language model's geometry;
2. whether language models agree with each other more than they agree with
   fastText;
3. the partial correlation between two language models with the fastText
   geometry regressed out -- if that collapses, "convergence" is just both
   models having learned standard distributional semantics.

Only concepts fastText actually covers are used, and coverage is reported,
because a missing word is a limit on the baseline rather than a detail.
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


def rank01(v):
    r = rankdata(v, method="average")
    return (r - r.mean()) / max(r.std(), 1e-12)


def partial(a, b, c, rng, n_perm):
    """Spearman(a, b) with c regressed out, plus a permutation null."""
    ra, rb, rc = rank01(a), rank01(b), rank01(c)
    X = np.column_stack([np.ones(len(rc)), rc])
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    obs = float(np.corrcoef(ea, eb)[0, 1])
    null = np.array([np.corrcoef(ea, rng.permutation(eb))[0, 1] for _ in range(n_perm)])
    return obs, float((1 + np.sum(null >= obs)) / (n_perm + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repr-dir", default="outputs/representations")
    ap.add_argument("--analysis-dir", default="outputs/analysis")
    ap.add_argument("--fasttext-dir", default="data/fasttext")
    ap.add_argument("--out", default="outputs/analysis/static_baseline.csv")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    with (ROOT / "benchmark/concepts.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["concept_id"] for r in rows]

    summary = json.loads((ROOT / args.analysis_dir / "summary.json").read_text())
    chosen = summary["selected_layers"]
    split = pd.read_csv(ROOT / args.analysis_dir / "concept_split.csv")
    eval_idx = np.array([i for i, r in enumerate(split.itertuples())
                         if r.split == "evaluation"], dtype=int)

    ft, covered = {}, {}
    for lang in ("en", "zh"):
        z = np.load(ROOT / args.fasttext_dir / f"{lang}.npz", allow_pickle=False)
        ft[lang] = z["vectors"]
        covered[lang] = z["present"]

    # Held-out concepts fastText covers in BOTH languages, so every comparison
    # below is over the same concept set.
    keep = np.array([i for i in eval_idx if covered["en"][i] and covered["zh"][i]], dtype=int)
    print(f"held-out concepts {len(eval_idx)}; fastText covers "
          f"en {int(covered['en'][eval_idx].sum())}, zh {int(covered['zh'][eval_idx].sum())}, "
          f"both {len(keep)} -- all comparisons use those {len(keep)}")

    systems = {}
    for path in sorted((ROOT / args.repr_dir).glob("*__bare.npz")):
        z = np.load(path, allow_pickle=False)
        key = f"{z['model_key']}|{z['language']}"
        if key in chosen:
            systems[key] = cosine_rdm(z["reps"].astype(np.float32)[:, chosen[key], :])
    for lang in ("en", "zh"):
        systems[f"fasttext|{lang}"] = cosine_rdm(ft[lang])

    vecs = {s: tri(d, keep) for s, d in systems.items()}
    llms = sorted(s for s in systems if not s.startswith("fasttext"))

    out = []
    for a, b in itertools.combinations(sorted(systems), 2):
        kind = ("fasttext-vs-llm" if a.startswith("fasttext") ^ b.startswith("fasttext")
                else "fasttext-internal" if a.startswith("fasttext") else "llm-vs-llm")
        row = {"system_a": a, "system_b": b, "kind": kind,
               "spearman": float(spearmanr(vecs[a], vecs[b]).statistic)}
        if kind == "llm-vs-llm":
            # Control with the fastText geometry of a's language.
            ctrl = vecs[f"fasttext|{a.split('|')[1]}"]
            row["partial_vs_fasttext"], row["partial_p"] = partial(
                vecs[a], vecs[b], ctrl, rng, args.permutations)
        out.append(row)

    df = pd.DataFrame(out)
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    for kind in ("llm-vs-llm", "fasttext-vs-llm", "fasttext-internal"):
        sub = df[df.kind == kind]
        if len(sub):
            print(f"\n{kind}: n={len(sub)} mean rho {sub.spearman.mean():.4f} "
                  f"[{sub.spearman.min():.4f}, {sub.spearman.max():.4f}]")
    ll = df[df.kind == "llm-vs-llm"]
    print(f"\nLLM-LLM agreement with fastText partialled out: "
          f"mean {ll.partial_vs_fasttext.mean():.4f} (raw {ll.spearman.mean():.4f}), "
          f"{int((ll.partial_p <= 0.05).sum())}/{len(ll)} significant at p<=.05")
    print(f"\n{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
