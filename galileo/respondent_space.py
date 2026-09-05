#!/usr/bin/env python3
"""Woelfel's actual move: a space whose points are respondents, not concepts.

Galileo does not stop at one map of concepts. Its analytical payload is the
comparison *between* respondents -- Woelfel's Bush voters and Dukakis voters each
carry a space, and the finding is how those spaces differ and how objects move
between them. This builds that second order.

Each respondent here is a whole population: a language model, or a group of
people. Each produces its own distance matrix over the *same* ten concepts. The
distance between two respondents is then how much they disagree about those
concepts, and metric MDS over that gives a map in which one point is one entire
worldview.

**This is why a 2-D map is defensible here when it was not before.** The concept
space genuinely needs about ten dimensions -- classical MDS retains only ~31% of
it in two -- so a flat map of concepts is mostly projection artefact, which is
why `make_visual.py` refuses to draw one. A respondent space has a handful of
points and far less intrinsic structure, so it can often be shown flat without
lying. That is not assumed: the retained variance is computed and printed, and
the figure says what it is.

Two properties are drawn rather than asserted:

* **Measurement error, as area.** Each respondent's corpus is split in half and
  each half placed independently. The distance between those two placements is
  how far a point moves for sampling reasons alone. A gap between respondents
  smaller than their blobs is not a difference.
* **The ceiling, as the same units.** Split-half agreement is Spearman-Brown
  corrected, so it bounds full-corpus comparisons rather than half-sized ones.
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
                        distance_matrix, spearman, spearman_brown, tokenize)
from compare_arms import content_tokens, take_tokens

ROOT = pathlib.Path(__file__).resolve().parent


def classical_mds(D, dim=2):
    """Torgerson MDS, returning coordinates and the fraction of structure kept."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w_all = np.maximum(w[order], 0)
    idx = order[:dim]
    coords = V[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    kept = w_all[:dim].sum() / w_all.sum() if w_all.sum() > 0 else float("nan")
    return coords, float(kept)


def procrustes(A, B):
    """Rotate/reflect/scale B onto A, so two independent solutions are comparable."""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, s, Vt = np.linalg.svd(B0.T @ A0)
    scale = s.sum() / max((B0 ** 2).sum(), 1e-12)
    return B0 @ (U @ Vt) * scale + A.mean(0)


def human_groups(shard, rng, n_groups=3):
    """Split people by sentiment -- the analogue of Woelfel's opposed voter groups.

    Splitting by topic instead would hand each group a different vocabulary and
    the result would only say that different people discuss different things.
    Sentiment holds the subject matter fixed and varies the stance toward it.
    """
    posts = [p for p in load_posts(shard, theme="Politics") if p["sentiment"] is not None]
    posts.sort(key=lambda p: p["sentiment"])
    cut = len(posts) // n_groups
    names = ({0: "people (hostile)", 1: "people (neutral)", 2: "people (warm)"}
             if n_groups == 3 else
             {0: "most hostile", 1: "hostile", 2: "neutral", 3: "warm", 4: "warmest"}
             if n_groups == 5 else {})
    out = {}
    for i in range(n_groups):
        chunk = posts[i * cut:(i + 1) * cut] if i < n_groups - 1 else posts[i * cut:]
        rng.shuffle(chunk)
        out[names.get(i, f"people {i}")] = chunk
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard",
                    default=str(pathlib.Path.home() / "projects/data/exorde/data_49.parquet"))
    ap.add_argument("--corpus", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--human-groups", type=int, default=3)
    ap.add_argument("--vocab-size", type=int, default=10)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--max-context", type=int, default=3000)
    ap.add_argument("--out", default="results/respondent_space.json")
    args = ap.parse_args()

    rng = random.Random(20260905)
    corpora = human_groups(args.shard, rng, args.human_groups)
    # Recorded explicitly rather than inferred downstream from the name. A
    # consumer guessing "is this a model?" from a name prefix silently
    # misclassified every human band the moment the labels changed.
    human_names = list(corpora)
    for spec in args.corpus:
        name, _, path = spec.partition("=")
        rows = [json.loads(l) for l in (ROOT / path).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        rng.shuffle(rows)
        corpora[name] = rows

    for k, v in corpora.items():
        print(f"{k:<18s} {len(v):>7,} posts  {content_tokens(v):>8,} content tokens")

    budget = min(content_tokens(v) for v in corpora.values())
    corpora = {k: take_tokens(v, budget) for k, v in corpora.items()}
    print(f"\nall {len(corpora)} respondents cut to {budget:,} content tokens")

    names = list(corpora)
    vocab = shared_vocabulary(list(corpora.values()), args.vocab_size)
    if len(vocab) < args.vocab_size:
        raise RuntimeError(f"only {len(vocab)} concepts are frequent in every respondent")
    print(f"shared concepts: {', '.join(vocab)}\n")

    iu = np.triu_indices(len(vocab), 1)

    def space(posts):
        return distance_matrix(ppmi_profiles(posts, vocab, args.window,
                                            max_context=args.max_context))

    RDM = {k: space(v) for k, v in corpora.items()}

    # Each respondent split in half and placed twice, so the map can show how far
    # a point drifts for sampling reasons alone.
    halves, ceiling = {}, {}
    for k, v in corpora.items():
        g = list(v); rng.shuffle(g); mid = len(g) // 2
        a, b = space(g[:mid]), space(g[mid:])
        halves[k] = (a, b)
        ceiling[k] = spearman_brown(spearman(a[iu], b[iu]))

    def disagreement(A, B):
        """1 - rank agreement: 0 when two respondents order concept pairs alike."""
        return 1.0 - spearman(A[iu], B[iu])

    n = len(names)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = disagreement(RDM[names[i]], RDM[names[j]])

    coords, kept = classical_mds(D)
    print(f"2-D map keeps {kept:.0%} of the structure among respondents")
    print("  (the CONCEPT space keeps only ~31%, which is why concepts are not "
          "mapped flat)\n")

    # Place each half in the same frame to get a per-respondent error radius.
    err = {}
    for k in names:
        a, b = halves[k]
        Dh = np.zeros((n + 1, n + 1))
        base = [RDM[m] for m in names]
        for i, j in itertools.combinations(range(n), 2):
            Dh[i, j] = Dh[j, i] = D[i, j]
        for half_i, H in ((n, a),):
            for i, M in enumerate(base):
                Dh[half_i, i] = Dh[i, half_i] = disagreement(M, H)
        ca, _ = classical_mds(Dh)
        Dh2 = Dh.copy()
        for i, M in enumerate(base):
            Dh2[n, i] = Dh2[i, n] = disagreement(M, b)
        cb, _ = classical_mds(Dh2)
        cb = procrustes(ca[:n], cb[:n]) if False else cb
        err[k] = float(np.linalg.norm(ca[n] - cb[n]))

    print(f"{'respondent':<18s} {'agrees w/ itself':>17s} {'point moves':>13s}")
    for k in names:
        print(f"{k:<18s} {ceiling[k]:>17.3f} {err[k]:>13.3f}")

    print(f"\ndisagreement between respondents (0 = identical ordering):")
    w = max(len(x) for x in names) + 1
    print(" " * (w + 1) + "".join(f"{x[:9]:>10s}" for x in names))
    for i, a in enumerate(names):
        print(f"{a:<{w}s} " + "".join(
            f"{'   --  ':>10s}" if i == j else f"{D[i, j]:>10.3f}"
            for j in range(n)))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "respondents": names, "human": human_names, "vocab": vocab,
        "matched_content_tokens": budget,
        "n_posts": {k: len(v) for k, v in corpora.items()},
        "disagreement": D.tolist(),
        "coords": coords.tolist(), "variance_kept_2d": kept,
        "ceiling": {k: float(v) for k, v in ceiling.items()},
        "error_radius": {k: float(v) for k, v in err.items()},
        "concept_distances": {k: v.tolist() for k, v in RDM.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
