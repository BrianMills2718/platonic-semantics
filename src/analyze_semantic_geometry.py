#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def cosine_rdm(x):
    x = np.asarray(x, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


def tri_vec(d):
    return d[np.triu_indices(len(d), 1)]


def sub_rdm(d, idx):
    idx = np.asarray(idx, dtype=int)
    return d[np.ix_(idx, idx)]


def rank01(v):
    r = rankdata(v, method="average")
    return ((r - 1) / max(1, len(r) - 1)).astype(np.float32)


def corr(a, b):
    x = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(x) else float(x)


def classical_mds(d, ndim=2):
    n = len(d)
    d2 = np.square(d.astype(np.float64))
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    coords = vecs[:, :ndim] * np.sqrt(np.maximum(vals[:ndim], 0))
    return coords.astype(np.float32), vals


def knn_sets(d, k):
    n = len(d)
    k = min(max(1, k), n - 1)
    out = []
    for i in range(n):
        idx = [int(j) for j in np.argsort(d[i]) if int(j) != i][:k]
        out.append(set(idx))
    return out


def mean_pairwise_jaccard(sets_by_system):
    systems = list(sets_by_system)
    n = len(next(iter(sets_by_system.values())))
    scores = np.zeros(n, np.float32)
    count = 0
    for a, b in itertools.combinations(systems, 2):
        count += 1
        for i in range(n):
            A, B = sets_by_system[a][i], sets_by_system[b][i]
            scores[i] += len(A & B) / max(1, len(A | B))
    return scores / max(1, count)


def select_layers(layer_rdms, selection_idx, iterations=5):
    """Select one layer/system using ONLY selection concepts."""
    systems = list(layer_rdms)
    chosen = {s: len(layer_rdms[s]) - 1 for s in systems}
    for _ in range(iterations):
        changed = False
        for s in systems:
            others = [o for o in systems if o != s]
            bestp = chosen[s]
            best = -1e9
            for p, d in enumerate(layer_rdms[s]):
                v = tri_vec(sub_rdm(d, selection_idx))
                score = np.mean([
                    corr(v, tri_vec(sub_rdm(layer_rdms[o][chosen[o]], selection_idx)))
                    for o in others
                ])
                if score > best:
                    best = score
                    bestp = p
            if bestp != chosen[s]:
                chosen[s] = bestp
                changed = True
        if not changed:
            break
    return chosen


def relation_signature(d, a, b):
    # Coordinate-free transformation: how distances to every anchor concept
    # change when moving from source a to target b.
    sig = (d[b] - d[a]).astype(np.float32)
    sig[a] = np.nan
    sig[b] = np.nan
    return sig


def masked_cos(a, b, anchor_mask=None):
    m = np.isfinite(a) & np.isfinite(b)
    if anchor_mask is not None:
        m &= anchor_mask
    if m.sum() < 3:
        return np.nan
    x = a[m] - np.mean(a[m])
    y = b[m] - np.mean(b[m])
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return np.nan
    return float(np.dot(x, y) / (nx * ny))


def mean_signature(sigs):
    arr = np.stack(sigs).astype(np.float32)
    valid = np.isfinite(arr)
    count = valid.sum(axis=0)
    total = np.where(valid, arr, 0.0).sum(axis=0)
    out = np.full(arr.shape[1], np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def loo_consistency(sigs, anchor_mask=None):
    if len(sigs) < 3:
        return np.nan
    vals = []
    for i, s in enumerate(sigs):
        mu = mean_signature([x for j, x in enumerate(sigs) if j != i])
        c = masked_cos(s, mu, anchor_mask)
        if np.isfinite(c):
            vals.append(c)
    return float(np.mean(vals)) if vals else np.nan


def relation_retrieval(d, pairs, candidate_idx=None, anchor_mask=None):
    # Leave-one-pair-out relation signature; rank candidate concepts as targets.
    if len(pairs) < 3:
        return (np.nan, np.nan, np.nan)
    n = len(d)
    candidates = list(range(n)) if candidate_idx is None else list(candidate_idx)
    ranks = []
    for i, (a, b) in enumerate(pairs):
        train = [relation_signature(d, x, y) for j, (x, y) in enumerate(pairs) if j != i]
        mu = mean_signature(train)
        scores = []
        for cand in candidates:
            if cand == a:
                continue
            s = relation_signature(d, a, cand)
            sc = masked_cos(s, mu, anchor_mask)
            scores.append((-999.0 if not np.isfinite(sc) else sc, cand))
        scores.sort(reverse=True)
        rank = next((r + 1 for r, (_, cand) in enumerate(scores) if cand == b), len(scores) + 1)
        ranks.append(rank)
    ranks = np.asarray(ranks, float)
    return float(np.mean(1 / ranks)), float(np.mean(ranks <= 1)), float(np.mean(ranks <= 10))


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_split(concept_rows, frac, seed):
    """Deterministic region-stratified split for layer selection vs evaluation."""
    rng = np.random.default_rng(seed)
    by_region = defaultdict(list)
    for i, row in enumerate(concept_rows):
        by_region[row["region"]].append(i)

    selection, evaluation = [], []
    for _, idxs in sorted(by_region.items()):
        idxs = np.asarray(idxs, dtype=int)
        rng.shuffle(idxs)
        if len(idxs) == 1:
            # Alternate singleton regions deterministically.
            (selection if len(selection) <= len(evaluation) else evaluation).extend(idxs.tolist())
            continue
        nsel = int(round(len(idxs) * frac))
        nsel = min(len(idxs) - 1, max(1, nsel))
        selection.extend(idxs[:nsel].tolist())
        evaluation.extend(idxs[nsel:].tolist())

    selection = np.asarray(sorted(selection), dtype=int)
    evaluation = np.asarray(sorted(evaluation), dtype=int)
    if len(selection) < 4 or len(evaluation) < 4:
        raise ValueError("Selection/evaluation split is too small.")
    return selection, evaluation


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.where(np.isfinite(p))[0]
    if not len(finite):
        return q
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    restored = np.empty(m)
    restored[order] = adj
    q[finite] = restored
    return q


def p_greater(observed, null):
    null = np.asarray([x for x in null if np.isfinite(x)], dtype=float)
    if not len(null) or not np.isfinite(observed):
        return np.nan
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def z_against_null(observed, null):
    null = np.asarray([x for x in null if np.isfinite(x)], dtype=float)
    if len(null) < 2 or not np.isfinite(observed):
        return np.nan
    sd = float(np.std(null, ddof=1))
    return np.nan if sd < 1e-12 else float((observed - np.mean(null)) / sd)


def bootstrap_mean_ci(values, rng, n_boot, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return (np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot[b] = np.mean(rng.choice(v, size=len(v), replace=True))
    return tuple(np.quantile(boot, [alpha / 2, 1 - alpha / 2]).tolist())


def bootstrap_rdm_corr_ci(d1, d2, idx, rng, n_boot):
    """Descriptive bootstrap over held-out pairwise distances.

    Pairwise distances are not fully independent, so this CI is descriptive;
    the permutation p-value remains the primary inferential statistic.
    """
    a = tri_vec(sub_rdm(d1, idx))
    b = tri_vec(sub_rdm(d2, idx))
    n = len(a)
    if n < 4:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        take = rng.integers(0, n, size=n)
        vals.append(corr(a[take], b[take]))
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def bootstrap_relation_convergence(by_system_sigs, rng, n_boot, anchor_mask):
    systems = list(by_system_sigs)
    n_pairs = min(len(by_system_sigs[s]) for s in systems)
    if n_pairs < 3:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        means = {}
        for s in systems:
            means[s] = mean_signature([by_system_sigs[s][i] for i in idx])
        pairvals = []
        for a, b in itertools.combinations(systems, 2):
            c = masked_cos(means[a], means[b], anchor_mask)
            if np.isfinite(c):
                pairvals.append(c)
        if pairvals:
            vals.append(float(np.mean(pairvals)))
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def target_reuse_ratio(pairs):
    """Pairs per distinct target. 1.0 means every pair has its own target."""
    if not pairs:
        return float("nan")
    return len(pairs) / len(set(int(b) for _, b in pairs))


def random_target_pairs(pairs, candidate_idx, rng, concept_regions=None, region_matched=False):
    """Break target identity while preserving source order, pair count, and the
    MULTIPLICITY PATTERN of the observed targets.

    Preserving multiplicity is not cosmetic. Resampling every pair's target
    independently destroys the target reuse present in the observed probes
    (e.g. an IsA set where seven pairs all end at `mammal` shares the whole
    d(mammal, .) term across those pairs, and the null does not). That makes the
    null strictly easier than the observed data for any relation that reuses
    targets, and turns the test into a target-reuse detector rather than a
    semantics detector: on pure Gaussian random geometry the previous version
    flagged such relations at p <= .05 in 12/12 runs, surviving BH-FDR in 11/12,
    at a rate monotone in the reuse ratio.

    A replacement target is therefore drawn once per DISTINCT observed target and
    reused for every pair that shared it. `tests/test_null_calibration.py` locks
    this in.

    With region_matched=True, each replacement is drawn from the same broad
    semantic region as the target it replaces whenever an alternative exists.
    """
    candidates = np.asarray(candidate_idx, dtype=int)
    # Sources sharing an observed target must keep sharing their replacement, so
    # the replacement is chosen once per group and must avoid every source in it
    # (otherwise the self-pair guard would split the group and break multiplicity).
    group_sources = {}
    for a, b in pairs:
        group_sources.setdefault(int(b), set()).add(int(a))

    mapping, used = {}, set()
    for b, sources in group_sources.items():
        blocked = used | sources | {b}
        pool = candidates
        if region_matched and concept_regions is not None:
            same = candidates[np.asarray([concept_regions[int(x)] == concept_regions[b] for x in candidates])]
            if len(same):
                pool = same
        valid = np.asarray([int(x) for x in pool if int(x) not in blocked], dtype=int)
        if not len(valid):
            valid = np.asarray([int(x) for x in candidates if int(x) not in blocked], dtype=int)
        if not len(valid):
            raise ValueError(
                f"No replacement target available for concept {b}; the candidate pool "
                f"is too small for a multiplicity-preserving null."
            )
        pick = int(rng.choice(valid))
        mapping[b] = pick
        used.add(pick)

    return [(int(a), mapping[int(b)]) for a, b in pairs]


def permuted_pairing_pairs(pairs, rng, tries=32):
    """Strictest relation null: keep the observed source multiset AND the observed
    target multiset exactly, and destroy only which source goes with which target.

    Used for within-system leave-one-out consistency. It is deliberately NOT used
    for cross-system mean-signature convergence, where the mean of
    d(b, .) - d(a, .) over a relation is very nearly invariant under a pairing
    permutation, so the null would collapse onto the observed value and the test
    would have no power by construction.
    """
    srcs = [int(a) for a, _ in pairs]
    tgts = [int(b) for _, b in pairs]
    for _ in range(tries):
        perm = rng.permutation(len(tgts))
        out = [(srcs[i], tgts[perm[i]]) for i in range(len(srcs))]
        if all(a != b for a, b in out):
            return out
    return [(a, b) for a, b in zip(srcs, tgts[1:] + tgts[:1]) if True]

def system_alignment_permutation(d1, d2, eval_idx, rng, n_perm):
    A = sub_rdm(d1, eval_idx)
    B = sub_rdm(d2, eval_idx)
    observed = corr(tri_vec(A), tri_vec(B))
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        p = rng.permutation(len(eval_idx))
        null[i] = corr(tri_vec(A), tri_vec(B[np.ix_(p, p)]))
    return observed, null


def neighborhood_null(selected_eval, k, rng, n_perm):
    systems = list(selected_eval)
    observed_knn = {s: knn_sets(selected_eval[s], k) for s in systems}
    observed_scores = mean_pairwise_jaccard(observed_knn)
    observed = float(np.mean(observed_scores))

    null = np.empty(n_perm, dtype=float)
    n = len(next(iter(selected_eval.values())))
    for b in range(n_perm):
        perm_knn = {}
        # Keep first system fixed; independently relabel all others.
        for si, s in enumerate(systems):
            d = selected_eval[s]
            if si == 0:
                dp = d
            else:
                p = rng.permutation(n)
                dp = d[np.ix_(p, p)]
            perm_knn[s] = knn_sets(dp, k)
        null[b] = float(np.mean(mean_pairwise_jaccard(perm_knn)))
    return observed_scores, observed, null


def relation_cross_system_score(sigs_by_system, anchor_mask):
    means = {s: mean_signature(v) for s, v in sigs_by_system.items()}
    vals = []
    for a, b in itertools.combinations(means, 2):
        c = masked_cos(means[a], means[b], anchor_mask)
        if np.isfinite(c):
            vals.append(c)
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser(description="Analyze shared semantic geometry with held-out and null tests.")
    ap.add_argument("--repr-dir", default="outputs/representations")
    ap.add_argument("--concepts", default="benchmark/concepts.csv")
    ap.add_argument("--relations", default="benchmark/relations.csv")
    ap.add_argument("--prompt-mode", default="bare")
    ap.add_argument("--outdir", default="outputs/analysis")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--selection-frac", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--permutations", type=int, default=250)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()

    if not (0.2 <= args.selection_frac <= 0.8):
        raise SystemExit("--selection-frac must be between 0.2 and 0.8")
    if args.permutations < 10:
        raise SystemExit("--permutations must be >= 10")
    if args.bootstrap < 20:
        raise SystemExit("--bootstrap must be >= 20")

    rng = np.random.default_rng(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    concept_rows = read_csv(args.concepts)
    ids = [r["concept_id"] for r in concept_rows]
    regions = {r["concept_id"]: r["region"] for r in concept_rows}
    index = {cid: i for i, cid in enumerate(ids)}
    rel_rows = read_csv(args.relations)
    selection_idx, eval_idx = stratified_split(concept_rows, args.selection_frac, args.seed)
    eval_mask = np.zeros(len(ids), dtype=bool)
    eval_mask[eval_idx] = True

    split_df = pd.DataFrame({
        "concept_id": ids,
        "region": [r["region"] for r in concept_rows],
        "split": ["selection" if i in set(selection_idx.tolist()) else "evaluation" for i in range(len(ids))]
    })
    split_df.to_csv(out / "concept_split.csv", index=False)

    files = sorted(Path(args.repr_dir).glob(f"*__*__{args.prompt_mode}.npz"))
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 representation files in {args.repr_dir}; found {len(files)}")

    reps, meta = {}, {}
    for p in files:
        z = np.load(p, allow_pickle=False)
        zids = [str(x) for x in z["concept_ids"].tolist()]
        if zids != ids:
            raise ValueError(f"Concept ordering mismatch: {p}")
        key = f"{str(z['model_key'])}|{str(z['language'])}"
        reps[key] = z["reps"].astype(np.float32)
        meta[key] = {"model": str(z["model_key"]), "language": str(z["language"])}
        print(key, reps[key].shape)

    # Every layer's representational distance geometry.
    layer_rdms = {
        s: [cosine_rdm(x[:, li, :]) for li in range(x.shape[1])]
        for s, x in reps.items()
    }

    # Layer selection sees ONLY the selection concepts.
    chosen = select_layers(layer_rdms, selection_idx)
    selected = {s: layer_rdms[s][chosen[s]] for s in reps}

    # Primary pairwise alignment is HELD OUT.
    align_rows = []
    for a, b in itertools.combinations(selected, 2):
        observed, null = system_alignment_permutation(
            selected[a], selected[b], eval_idx, rng, args.permutations
        )
        lo, hi = bootstrap_rdm_corr_ci(
            selected[a], selected[b], eval_idx, rng, args.bootstrap
        )
        align_rows.append({
            "system_a": a, "system_b": b,
            "heldout_spearman_rho": observed,
            "null_mean": float(np.mean(null)),
            "null_sd": float(np.std(null, ddof=1)),
            "effect_over_null": float(observed - np.mean(null)),
            "z_vs_null": z_against_null(observed, null),
            "permutation_p": p_greater(observed, null),
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
        })
    align_df = pd.DataFrame(align_rows)
    align_df["fdr_q"] = bh_fdr(align_df["permutation_p"].to_numpy())
    align_df.to_csv(out / "system_alignment.csv", index=False)

    # For diagnostic comparison only: how good selection alignment looked on the selection split.
    train_rows = []
    for a, b in itertools.combinations(selected, 2):
        train_rows.append({
            "system_a": a, "system_b": b,
            "selection_spearman_rho": corr(
                tri_vec(sub_rdm(selected[a], selection_idx)),
                tri_vec(sub_rdm(selected[b], selection_idx))
            )
        })
    pd.DataFrame(train_rows).to_csv(out / "system_alignment_selection.csv", index=False)

    # Layer-level English/Chinese curves, separately on selection and held-out concepts.
    curves = []
    models = sorted(set(v["model"] for v in meta.values()))
    for m in models:
        en, zh = f"{m}|en", f"{m}|zh"
        if en not in layer_rdms or zh not in layer_rdms:
            continue
        A, B = layer_rdms[en], layer_rdms[zh]
        L = min(len(A), len(B))
        for i in range(L):
            ia = round(i * (len(A) - 1) / max(1, L - 1))
            ib = round(i * (len(B) - 1) / max(1, L - 1))
            curves.append({
                "model": m, "relative_layer": i / max(1, L - 1),
                "en_layer": ia, "zh_layer": ib,
                "selection_rdm_spearman": corr(
                    tri_vec(sub_rdm(A[ia], selection_idx)),
                    tri_vec(sub_rdm(B[ib], selection_idx))
                ),
                "heldout_rdm_spearman": corr(
                    tri_vec(sub_rdm(A[ia], eval_idx)),
                    tri_vec(sub_rdm(B[ib], eval_idx))
                )
            })
    pd.DataFrame(curves).to_csv(out / "cross_language_layer_curve.csv", index=False)

    # Consensus uses all concepts ONLY AFTER layers are fixed.
    tri = np.triu_indices(len(ids), 1)
    rv = np.stack([rank01(d[tri]) for d in selected.values()])
    cv = rv.mean(0)
    consensus = np.zeros((len(ids), len(ids)), np.float32)
    consensus[tri] = cv
    consensus[(tri[1], tri[0])] = cv
    coords, eig = classical_mds(consensus)

    # Held-out neighborhood stability + identity-destroying null.
    selected_eval = {s: sub_rdm(d, eval_idx) for s, d in selected.items()}
    heldout_scores, heldout_mean, heldout_null = neighborhood_null(
        selected_eval, args.knn, rng, args.permutations
    )
    nh_lo, nh_hi = bootstrap_mean_ci(heldout_scores, rng, args.bootstrap)
    neighborhood_test = pd.DataFrame([{
        "metric": "mean_cross_system_knn_jaccard",
        "heldout_observed": heldout_mean,
        "bootstrap_ci_lo": nh_lo,
        "bootstrap_ci_hi": nh_hi,
        "null_mean": float(np.mean(heldout_null)),
        "null_sd": float(np.std(heldout_null, ddof=1)),
        "effect_over_null": float(heldout_mean - np.mean(heldout_null)),
        "z_vs_null": z_against_null(heldout_mean, heldout_null),
        "permutation_p": p_greater(heldout_mean, heldout_null),
    }])
    neighborhood_test.to_csv(out / "neighborhood_null_test.csv", index=False)

    # Full-map descriptive stability after layer choice.
    knn_full = {s: knn_sets(d, args.knn) for s, d in selected.items()}
    global_stab = mean_pairwise_jaccard(knn_full)
    concept_out = []
    for i, cid in enumerate(ids):
        ms = []
        for lang in sorted(set(v["language"] for v in meta.values())):
            ss = [s for s in knn_full if meta[s]["language"] == lang]
            if len(ss) >= 2:
                vals = []
                for a, b in itertools.combinations(ss, 2):
                    A, B = knn_full[a][i], knn_full[b][i]
                    vals.append(len(A & B) / max(1, len(A | B)))
                ms.append(np.mean(vals))
        ls = []
        for model in models:
            ss = [s for s in knn_full if meta[s]["model"] == model]
            if len(ss) >= 2:
                vals = []
                for a, b in itertools.combinations(ss, 2):
                    A, B = knn_full[a][i], knn_full[b][i]
                    vals.append(len(A & B) / max(1, len(A | B)))
                ls.append(np.mean(vals))
        concept_out.append({
            "concept_id": cid, "en": concept_rows[i]["en"], "zh": concept_rows[i]["zh"],
            "region": regions[cid], "split": "selection" if i in set(selection_idx.tolist()) else "evaluation",
            "x": coords[i, 0], "y": coords[i, 1],
            "global_stability": global_stab[i],
            "model_stability": float(np.mean(ms)) if ms else np.nan,
            "language_stability": float(np.mean(ls)) if ls else np.nan,
        })
    pd.DataFrame(concept_out).to_csv(out / "consensus_coordinates.csv", index=False)

    # Relations: all labeled pairs are retained for power, but relation signatures are
    # compared only on HELD-OUT ANCHOR dimensions. Layer selection never saw distances
    # involving those evaluation concepts.
    relpairs_all = defaultdict(list)
    for r in rel_rows:
        if r["source"] in index and r["target"] in index:
            relpairs_all[r["relation"]].append((index[r["source"]], index[r["target"]]))

    concept_regions_idx = [concept_rows[i]["region"] for i in range(len(ids))]
    relation_rows = []
    means_all = defaultdict(dict)
    sigs_anchor_eval = defaultdict(dict)

    for rel, pairs in sorted(relpairs_all.items()):
        for s, d in selected.items():
            sigs = [relation_signature(d, a, b) for a, b in pairs]
            means_all[rel][s] = mean_signature(sigs)
            sigs_anchor_eval[rel][s] = sigs

            loo = loo_consistency(sigs, eval_mask)
            mrr, h1, h10 = relation_retrieval(
                d, pairs, candidate_idx=np.arange(len(ids)), anchor_mask=eval_mask
            )

            # Two nulls:
            # 1) global random target
            # 2) harder coarse-semantic-region-matched random target
            null_global, null_region, null_pairing = [], [], []
            for _ in range(args.permutations):
                gpairs = random_target_pairs(
                    pairs, np.arange(len(ids)), rng,
                    concept_regions=concept_regions_idx, region_matched=False
                )
                rpairs = random_target_pairs(
                    pairs, np.arange(len(ids)), rng,
                    concept_regions=concept_regions_idx, region_matched=True
                )
                ppairs = permuted_pairing_pairs(pairs, rng)
                gsigs = [relation_signature(d, a, b) for a, b in gpairs]
                rsigs = [relation_signature(d, a, b) for a, b in rpairs]
                psigs = [relation_signature(d, a, b) for a, b in ppairs]
                null_global.append(loo_consistency(gsigs, eval_mask))
                null_region.append(loo_consistency(rsigs, eval_mask))
                null_pairing.append(loo_consistency(psigs, eval_mask))

            relation_rows.append({
                "relation": rel, "system": s, "n_pairs": len(pairs),
                "target_reuse_ratio": target_reuse_ratio(pairs),
                "heldout_anchor_signature_loo_consistency": loo,
                "global_null_mean": float(np.nanmean(null_global)),
                "global_effect_over_null": float(loo - np.nanmean(null_global)) if np.isfinite(loo) else np.nan,
                "global_permutation_p": p_greater(loo, null_global),
                "region_matched_null_mean": float(np.nanmean(null_region)),
                "region_matched_effect_over_null": float(loo - np.nanmean(null_region)) if np.isfinite(loo) else np.nan,
                "region_matched_permutation_p": p_greater(loo, null_region),
                "permuted_pairing_null_mean": float(np.nanmean(null_pairing)),
                "permuted_pairing_effect_over_null": float(loo - np.nanmean(null_pairing)) if np.isfinite(loo) else np.nan,
                "permuted_pairing_permutation_p": p_greater(loo, null_pairing),
                "target_mrr": mrr, "target_hit_at_1": h1, "target_hit_at_10": h10,
            })

    relation_df = pd.DataFrame(relation_rows)
    relation_df["global_fdr_q"] = bh_fdr(relation_df["global_permutation_p"].to_numpy())
    relation_df["region_matched_fdr_q"] = bh_fdr(relation_df["region_matched_permutation_p"].to_numpy())
    relation_df["permuted_pairing_fdr_q"] = bh_fdr(relation_df["permuted_pairing_permutation_p"].to_numpy())
    relation_df.to_csv(out / "relation_system_scores.csv", index=False)

    # Coordinate-free cross-system convergence:
    # relation pair can be any benchmark pair; similarity is computed only over held-out anchors.
    cross_rows = []
    for rel, pairs in sorted(relpairs_all.items()):
        by_system = sigs_anchor_eval[rel]
        if len(pairs) < 3 or len(by_system) < 2:
            continue

        observed = relation_cross_system_score(by_system, eval_mask)
        blo, bhi = bootstrap_relation_convergence(by_system, rng, args.bootstrap, eval_mask)

        null_global, null_region = [], []
        for _ in range(args.permutations):
            gpairs = random_target_pairs(
                pairs, np.arange(len(ids)), rng,
                concept_regions=concept_regions_idx, region_matched=False
            )
            rpairs = random_target_pairs(
                pairs, np.arange(len(ids)), rng,
                concept_regions=concept_regions_idx, region_matched=True
            )
            gbysystem = {
                s: [relation_signature(selected[s], a, b) for a, b in gpairs]
                for s in selected
            }
            rbysystem = {
                s: [relation_signature(selected[s], a, b) for a, b in rpairs]
                for s in selected
            }
            null_global.append(relation_cross_system_score(gbysystem, eval_mask))
            null_region.append(relation_cross_system_score(rbysystem, eval_mask))

        cross_rows.append({
            "relation": rel, "n_pairs": len(pairs),
            "target_reuse_ratio": target_reuse_ratio(pairs),
            "heldout_anchor_count": int(eval_mask.sum()),
            "systems": len(by_system),
            "heldout_anchor_cross_system_signature_similarity": observed,
            "bootstrap_ci_lo": blo, "bootstrap_ci_hi": bhi,
            "global_null_mean": float(np.nanmean(null_global)),
            "global_null_sd": float(np.nanstd(null_global, ddof=1)),
            "global_effect_over_null": float(observed - np.nanmean(null_global)),
            "global_z_vs_null": z_against_null(observed, null_global),
            "global_permutation_p": p_greater(observed, null_global),
            "region_matched_null_mean": float(np.nanmean(null_region)),
            "region_matched_null_sd": float(np.nanstd(null_region, ddof=1)),
            "region_matched_effect_over_null": float(observed - np.nanmean(null_region)),
            "region_matched_z_vs_null": z_against_null(observed, null_region),
            "region_matched_permutation_p": p_greater(observed, null_region),
        })

    cross_df = pd.DataFrame(cross_rows)
    cross_df["global_fdr_q"] = bh_fdr(cross_df["global_permutation_p"].to_numpy())
    cross_df["region_matched_fdr_q"] = bh_fdr(cross_df["region_matched_permutation_p"].to_numpy())
    cross_df.to_csv(out / "relation_convergence.csv", index=False)

    # Sensitivity: all-pairs relation convergence is descriptive, not the primary test.
    descript = []
    for rel, by_system in sorted(means_all.items()):
        vals = []
        for a, b in itertools.combinations(by_system, 2):
            c = masked_cos(by_system[a], by_system[b], None)
            if np.isfinite(c):
                vals.append(c)
        descript.append({
            "relation": rel,
            "all_pairs_cross_system_similarity": float(np.mean(vals)) if vals else np.nan,
            "min_pairwise_similarity": float(np.min(vals)) if vals else np.nan,
            "max_pairwise_similarity": float(np.max(vals)) if vals else np.nan,
        })
    pd.DataFrame(descript).to_csv(out / "relation_convergence_descriptive_all_concepts.csv", index=False)

    summary = {
        "systems": meta,
        "selected_layers": {s: int(chosen[s]) for s in chosen},
        "concepts": len(ids),
        "selection_concepts": int(len(selection_idx)),
        "evaluation_concepts": int(len(eval_idx)),
        "selection_fraction": args.selection_frac,
        "relations": len(rel_rows),
        "relation_types": sorted(relpairs_all),
        "knn": args.knn,
        "seed": args.seed,
        "permutations": args.permutations,
        "bootstrap_replicates": args.bootstrap,
        "heldout_mean_neighborhood_stability": heldout_mean,
        "heldout_neighborhood_permutation_p": float(neighborhood_test.iloc[0]["permutation_p"]),
        "consensus_projection_positive_eigenvalue_fraction_2d":
            float(np.maximum(eig[:2], 0).sum() / max(1e-12, np.maximum(eig, 0).sum())),
        "relation_target_reuse_ratio": {rel: target_reuse_ratio(p) for rel, p in sorted(relpairs_all.items())},
        "primary_tests": [
            "system_alignment.csv: held-out RDM alignment vs concept-identity permutation null",
            "neighborhood_null_test.csv: held-out cross-system kNN stability vs relabeled-system null",
            "relation_system_scores.csv: relation consistency measured on held-out anchor dimensions vs global and region-matched random-target nulls",
            "relation_convergence.csv: cross-system relation-signature convergence on held-out anchor dimensions vs global and region-matched nulls",
        ],
        "interpretation": [
            "Layer selection is performed only on the selection concept split.",
            "Headline convergence statistics are evaluated only on held-out concepts.",
            "Consensus display coordinates may use all concepts after layers are fixed, but 2D projection is not itself evidence.",
            "Permutation tests destroy the proposed correspondence while preserving the geometry of each system.",
            "Relation signatures compare distance changes only on held-out concept-anchor dimensions, not raw neuron coordinates.",
            "Relation nulls preserve the multiplicity pattern of the observed targets; a null that resamples targets independently measures target reuse, not semantics.",
            "The permuted-pairing null keeps both observed multisets exactly and is the strictest within-system relation baseline.",
            "Region-matched target nulls control for broad semantic-region membership.",
            "FDR q-values correct families of permutation tests using Benjamini-Hochberg."
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote hardened analysis to {out}")


if __name__ == "__main__":
    main()
