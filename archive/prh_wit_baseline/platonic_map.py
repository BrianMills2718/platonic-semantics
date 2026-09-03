#!/usr/bin/env python3
"""
Empirical "Platonic map" builder.

Constructs a 2D consensus geometry from real hidden-state features produced by
different neural models on the SAME stimuli.

Default data source:
  Huh, Cheung, Wang & Isola (ICML 2024), The Platonic Representation Hypothesis
  Precomputed WIT-1024 model features published by the authors.

Method:
  1. Load N shared stimuli from each model.
  2. For each layer, compute the representational dissimilarity matrix (RDM)
     using cosine distance.
  3. Pick one layer per model by coordinate ascent: choose the layer whose
     ranked RDM agrees most with the currently selected layers of other models.
  4. Rank-normalize each selected RDM and average them. This produces a
     "consensus geometry" without pretending the models use the same coordinates.
  5. Project the consensus distance matrix to 2D with classical MDS.
  6. Compute per-item cross-model k-nearest-neighbor agreement.
  7. Save an interactive HTML map, CSV coordinates, and JSON metadata.

This is an empirical estimate of shared representational geometry, not a proof
of a unique universal latent space.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import plotly.io as pio

FEATURES = {
    # language
    "bloom_560m": (
        "language",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "bigscience_bloomz-560m_pool-avg.pt",
    ),
    "bloom_1b1": (
        "language",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "bigscience_bloomz-1b1_pool-avg.pt",
    ),
    "bloom_1b7": (
        "language",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "bigscience_bloomz-1b7_pool-avg.pt",
    ),
    "openllama_3b": (
        "language",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "openlm-research_open_llama_3b_pool-avg.pt",
    ),
    # vision
    "i21k_t": (
        "vision",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "vit_tiny_patch16_224.augreg_in21k-cls.pt",
    ),
    "dinov2_s": (
        "vision",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "vit_small_patch14_dinov2.lvd142m_pool-cls.pt",
    ),
    "clip_b": (
        "vision",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "vit_base_patch16_clip_224.laion2b_pool-cls.pt",
    ),
    "clip_l": (
        "vision",
        "http://vision14.csail.mit.edu/prh/wit_1024/"
        "vit_large_patch14_clip_224.laion2b_pool-cls.pt",
    ),
}

DEFAULT_MODELS = ["bloom_560m", "bloom_1b1", "i21k_t", "dinov2_s", "clip_b"]


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024:
        print(f"[cached] {out.name}")
        return

    candidates = [url]
    if url.startswith("http://"):
        candidates.insert(0, "https://" + url[len("http://"):])

    last_error = None
    for candidate in candidates:
        try:
            print(f"[download] {candidate}")
            with requests.get(candidate, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", "0"))
                got = 0
                tmp = out.with_suffix(out.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            pct = 100 * got / total
                            print(f"\r  {pct:5.1f}%  {got/1e6:,.1f}/{total/1e6:,.1f} MB", end="")
                if total:
                    print()
                tmp.replace(out)
            return
        except Exception as exc:
            last_error = exc
            print(f"  failed: {exc}")
    raise RuntimeError(f"Could not download {url}: {last_error}")


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_feature_tensor(path: Path, n: int) -> np.ndarray:
    obj = safe_torch_load(path)
    if not isinstance(obj, dict) or "feats" not in obj:
        raise ValueError(f"{path} does not look like a PRH feature file.")
    feats = obj["feats"]

    if isinstance(feats, list):
        # Convert list[layer] of [N,D] to [N,L,D].
        feats = torch.stack(feats, dim=1)
    if not torch.is_tensor(feats):
        raise TypeError(f"Unsupported feature type in {path}: {type(feats)}")

    feats = feats[:n].float().cpu()
    if feats.ndim == 2:
        feats = feats[:, None, :]
    if feats.ndim != 3:
        raise ValueError(f"Expected [items,layers,dim], got {tuple(feats.shape)} in {path}")
    return feats.numpy()


def cosine_rdm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    sim = x @ x.T
    np.clip(sim, -1.0, 1.0, out=sim)
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32, copy=False)


def upper_tri_vector(d: np.ndarray, tri: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return d[tri].astype(np.float32, copy=False)


def rank_unit(v: np.ndarray) -> np.ndarray:
    r = rankdata(v, method="average").astype(np.float32)
    r -= r.mean()
    norm = np.linalg.norm(r)
    if norm < 1e-12:
        return np.zeros_like(r)
    return r / norm


def percentile_rank(v: np.ndarray) -> np.ndarray:
    r = rankdata(v, method="average").astype(np.float32)
    if len(r) <= 1:
        return np.zeros_like(r)
    return (r - 1.0) / (len(r) - 1.0)


def build_candidate_rdms(
    feats: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray],
    max_layers: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    n, layers, _ = feats.shape
    if max_layers and layers > max_layers:
        idxs = np.unique(np.linspace(0, layers - 1, max_layers).round().astype(int)).tolist()
    else:
        idxs = list(range(layers))

    raw_vectors, rank_vectors = [], []
    for li in idxs:
        d = cosine_rdm(feats[:, li, :])
        v = upper_tri_vector(d, tri)
        raw_vectors.append(v)
        rank_vectors.append(rank_unit(v))
    return raw_vectors, rank_vectors, idxs


def select_layers(
    rank_candidates: Dict[str, List[np.ndarray]],
    layer_indices: Dict[str, List[int]],
    iterations: int = 4,
) -> Dict[str, int]:
    models = list(rank_candidates)
    chosen_pos = {m: len(rank_candidates[m]) - 1 for m in models}

    for _ in range(iterations):
        changed = False
        for m in models:
            others = [o for o in models if o != m]
            if not others:
                continue
            best_pos = chosen_pos[m]
            best = -np.inf
            for p, rv in enumerate(rank_candidates[m]):
                scores = [float(np.dot(rv, rank_candidates[o][chosen_pos[o]])) for o in others]
                score = float(np.mean(scores))
                if score > best:
                    best = score
                    best_pos = p
            if best_pos != chosen_pos[m]:
                chosen_pos[m] = best_pos
                changed = True
        if not changed:
            break

    return {m: layer_indices[m][chosen_pos[m]] for m in models}


def selected_vector(
    model: str,
    selected_layer: int,
    raw_candidates: Dict[str, List[np.ndarray]],
    layer_indices: Dict[str, List[int]],
) -> np.ndarray:
    pos = layer_indices[model].index(selected_layer)
    return raw_candidates[model][pos]


def vector_to_matrix(v: np.ndarray, n: int, tri: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    d = np.zeros((n, n), dtype=np.float32)
    d[tri] = v
    d[(tri[1], tri[0])] = v
    return d


def classical_mds(d: np.ndarray, ndim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    n = d.shape[0]
    d2 = np.square(d.astype(np.float64))
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    positive = np.maximum(vals[:ndim], 0.0)
    coords = vecs[:, :ndim] * np.sqrt(positive)
    return coords.astype(np.float32), vals


def neighborhood_agreement(ds: List[np.ndarray], k: int) -> np.ndarray:
    n = ds[0].shape[0]
    k = max(2, min(k, n - 1))
    knns = []
    for d in ds:
        # first index is self, exclude it
        idx = np.argpartition(d, kth=k, axis=1)[:, : k + 1]
        rows = []
        for i in range(n):
            row = [int(x) for x in idx[i] if int(x) != i]
            row = sorted(row, key=lambda x: float(d[i, x]))[:k]
            rows.append(set(row))
        knns.append(rows)

    out = np.zeros(n, dtype=np.float32)
    pairs = 0
    for a in range(len(knns)):
        for b in range(a + 1, len(knns)):
            pairs += 1
            for i in range(n):
                A, B = knns[a][i], knns[b][i]
                union = len(A | B)
                out[i] += (len(A & B) / union) if union else 0.0
    if pairs:
        out /= pairs
    return out


def load_labels(n: int, labels_csv: str | None) -> List[str]:
    if labels_csv:
        p = Path(labels_csv)
        df = pd.read_csv(p)
        col = "label" if "label" in df.columns else df.columns[0]
        vals = df[col].astype(str).tolist()
        vals += [f"item {i}" for i in range(len(vals), n)]
        return vals[:n]

    try:
        from datasets import load_dataset
        print("[labels] loading minhuh/prh revision wit_1024 from Hugging Face")
        ds = load_dataset("minhuh/prh", revision="wit_1024", split="train")
        labels = []
        for row in ds.select(range(min(n, len(ds)))):
            text = row.get("text", "")
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            labels.append(str(text))
        labels += [f"item {i}" for i in range(len(labels), n)]
        return labels[:n]
    except Exception as exc:
        print(f"[labels] warning: dataset labels unavailable ({exc})")
        print("[labels] using item indices. You can supply --labels-csv labels.csv")
        return [f"item {i}" for i in range(n)]


def cluster_precomputed(d: np.ndarray, n_clusters: int) -> np.ndarray:
    n_clusters = max(2, min(n_clusters, len(d) - 1))
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
    except TypeError:
        # older scikit-learn
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )
    return model.fit_predict(d)


def make_html(
    df: pd.DataFrame,
    model_info: List[dict],
    out: Path,
    eigvals: np.ndarray,
) -> None:
    df = df.copy()
    df["cluster"] = df["cluster"].astype(str)
    df["agreement_pct"] = (100 * df["agreement"]).round(1)
    df["short_label"] = df["label"].str.replace(r"\s+", " ", regex=True).str.slice(0, 110)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        size="agreement",
        size_max=18,
        hover_name="short_label",
        hover_data={
            "item": True,
            "agreement_pct": True,
            "cluster": True,
            "x": ":.3f",
            "y": ":.3f",
            "agreement": False,
            "label": False,
        },
        labels={
            "x": "Consensus dimension 1",
            "y": "Consensus dimension 2",
            "agreement_pct": "kNN agreement (%)",
            "cluster": "Unsupervised cluster",
        },
        title="Empirical shared-representation map",
    )
    fig.update_traces(marker=dict(opacity=0.78, line=dict(width=0)))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Cluster",
        margin=dict(l=30, r=30, t=70, b=30),
        height=760,
    )

    plot = pio.to_html(fig, full_html=False, include_plotlyjs=True)
    model_rows = "\n".join(
        f"<tr><td>{html.escape(m['model'])}</td>"
        f"<td>{html.escape(m['modality'])}</td>"
        f"<td>{m['layer']}</td>"
        f"<td>{m['alignment_to_consensus']:.3f}</td></tr>"
        for m in model_info
    )
    pos = eigvals[eigvals > 0]
    explained = ""
    if len(pos) >= 2:
        explained = f"{100*pos[0]/pos.sum():.1f}% + {100*pos[1]/pos.sum():.1f}% of positive classical-MDS eigenvalue mass"

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Empirical Platonic Map</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px auto; max-width: 1180px; padding: 0 18px; color:#171717; }}
.note {{ background:#f5f5f5; border-left:4px solid #666; padding:12px 16px; margin:14px 0; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0 28px; }}
th,td {{ border-bottom:1px solid #ddd; text-align:left; padding:8px; }}
small {{ color:#666; }}
</style>
</head>
<body>
<h1>Empirical shared-representation map</h1>
<p>This is a projection of a <b>consensus distance geometry</b> across the selected real neural models.
It does not assume that their raw hidden coordinates are identical.</p>
<div class="note">
<b>How to read it:</b> nearby points are stimuli that the included models tend to place near one another.
Larger points have higher cross-model nearest-neighbor agreement. Colors are unsupervised clusters and
should not be interpreted as ground-truth concepts.
</div>
{plot}
<h2>Selected model layers</h2>
<table>
<tr><th>Model</th><th>Modality</th><th>Selected layer</th><th>Spearman alignment to consensus</th></tr>
{model_rows}
</table>
<p><small>2D projection diagnostic: {html.escape(explained)}. A 2D map necessarily loses information from
the high-dimensional consensus geometry. Inspect the agreement and model-alignment statistics before
interpreting local patterns.</small></p>
</body></html>"""
    out.write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=sorted(FEATURES),
                    help="PRH precomputed feature sets to include.")
    ap.add_argument("--items", type=int, default=384,
                    help="Number of shared WIT stimuli to map (max 1024).")
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--max-layers", type=int, default=0,
                    help="If >0, sample at most this many layers per model during layer selection.")
    ap.add_argument("--cache", default="./prh_feature_cache")
    ap.add_argument("--outdir", default="./platonic_map_output")
    ap.add_argument("--labels-csv", default=None,
                    help="Optional CSV with a 'label' column, used if HF labels are unavailable.")
    args = ap.parse_args()

    if len(args.models) < 2:
        ap.error("Use at least two models.")
    n = max(8, min(int(args.items), 1024))
    outdir = Path(args.outdir)
    cache = Path(args.cache)
    outdir.mkdir(parents=True, exist_ok=True)

    # Download + load
    features = {}
    modalities = {}
    for model in args.models:
        modality, url = FEATURES[model]
        modalities[model] = modality
        path = cache / (model + ".pt")
        download(url, path)
        print(f"[load] {model}")
        features[model] = load_feature_tensor(path, n)
        print(f"       shape={features[model].shape}")

    # Ensure all contain n items.
    n = min(x.shape[0] for x in features.values())
    for m in features:
        features[m] = features[m][:n]

    tri = np.triu_indices(n, k=1)
    raw_candidates: Dict[str, List[np.ndarray]] = {}
    rank_candidates: Dict[str, List[np.ndarray]] = {}
    layer_indices: Dict[str, List[int]] = {}

    for model, feats in features.items():
        print(f"[RDM] {model}: {feats.shape[1]} layers")
        raws, ranks, idxs = build_candidate_rdms(feats, tri, args.max_layers)
        raw_candidates[model] = raws
        rank_candidates[model] = ranks
        layer_indices[model] = idxs

    chosen = select_layers(rank_candidates, layer_indices)
    print("[layers]", chosen)

    selected_raw = {}
    selected_ds = {}
    for model in args.models:
        v = selected_vector(model, chosen[model], raw_candidates, layer_indices)
        selected_raw[model] = v
        selected_ds[model] = vector_to_matrix(v, n, tri)

    # Consensus distance = average percentile-ranked distance across models.
    consensus_vec = np.mean(
        np.stack([percentile_rank(selected_raw[m]) for m in args.models], axis=0),
        axis=0,
    ).astype(np.float32)
    consensus_d = vector_to_matrix(consensus_vec, n, tri)

    coords, eigvals = classical_mds(consensus_d, ndim=2)
    agreement = neighborhood_agreement([selected_ds[m] for m in args.models], args.knn)
    clusters = cluster_precomputed(consensus_d, args.clusters)
    labels = load_labels(n, args.labels_csv)

    df = pd.DataFrame({
        "item": np.arange(n),
        "label": labels,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "agreement": agreement,
        "cluster": clusters,
    })
    csv_path = outdir / "platonic_map_coordinates.csv"
    df.to_csv(csv_path, index=False)

    model_info = []
    for model in args.models:
        rho = float(spearmanr(selected_raw[model], consensus_vec).statistic)
        model_info.append({
            "model": model,
            "modality": modalities[model],
            "layer": int(chosen[model]),
            "alignment_to_consensus": rho,
        })

    meta = {
        "models": model_info,
        "items": n,
        "knn": args.knn,
        "clusters": args.clusters,
        "method": {
            "within_model_distance": "cosine distance",
            "layer_selection": "coordinate ascent maximizing cross-model ranked-RDM agreement",
            "consensus": "mean percentile-rank RDM",
            "projection": "classical MDS to 2D",
            "agreement": "mean pairwise kNN Jaccard across models",
        },
    }
    (outdir / "platonic_map_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    html_path = outdir / "platonic_map.html"
    make_html(df, model_info, html_path, eigvals)

    print("\nDONE")
    print(f"Interactive map: {html_path.resolve()}")
    print(f"Coordinates:     {csv_path.resolve()}")
    print(f"Metadata:        {(outdir / 'platonic_map_metadata.json').resolve()}")


if __name__ == "__main__":
    main()
