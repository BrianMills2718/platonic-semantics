#!/usr/bin/env python3
"""How much agreement is achievable at all? Run 001 reported rho without a denominator.

A held-out cross-system agreement of 0.27 means nothing on its own. It could be
0.27 out of an achievable 0.30, which would be near-perfect convergence, or 0.27
out of an achievable 0.95, which would be weak. Nothing in run 001 distinguishes
those, and every interpretation in RUN_001_RESULTS.md is silently assuming one.

This measures the ceiling the cheap way, from data already on disk. The same
model and language is extracted twice under different stimulus formulations --
the bare term, and a mean over six prompt templates with the term's own tokens
pooled -- and the correlation between those two geometries is how much of a
system's own concept organisation survives a change in how the concept was
presented. Cross-system agreement cannot exceed the geometric mean of the two
systems' reliabilities except by noise, so

    normalised = rho_AB / sqrt(r_AA * r_BB)

is agreement expressed as a fraction of what was reachable.

What this is and is not. It is a stimulus-formulation reliability: it asks
whether "dog" and "I am thinking about dog" put the same concept in the same
place. It is not a training-seed ceiling, which needs independently trained
models of the same architecture (Pythia publishes them) and remains the better
measurement. Treat this as a first bound, and read it as such: a low
reliability caps the headline, a high one does not license it.

Requires `<model>__<lang>__averaged.npz` beside the `__bare` files.

    python scripts/prompt_reliability_ceiling.py
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_semantic_geometry import (  # noqa: E402
    corr,
    cosine_rdm,
    stratified_split,
    sub_rdm,
    tri_vec,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repr-dir", default="outputs/representations")
    ap.add_argument("--analysis-dir", default="outputs/analysis")
    ap.add_argument("--out", default="outputs/analysis/prompt_reliability_ceiling.csv")
    ap.add_argument("--reliability-floor", type=float, default=0.30,
                    help="a system whose geometry survives a prompt change below this is "
                         "not a usable instrument and its pairs are reported separately")
    args = ap.parse_args()

    with (ROOT / "benchmark/concepts.csv").open(encoding="utf-8") as f:
        concept_rows = list(csv.DictReader(f))

    summary = json.loads((ROOT / args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
    layers = summary["selected_layers"]
    _, eval_idx = stratified_split(concept_rows, summary["selection_fraction"], summary["seed"])

    repr_dir = ROOT / args.repr_dir
    rdms, reliability = {}, {}
    for system, layer in sorted(layers.items()):
        model, lang = system.split("|")
        paths = {m: repr_dir / f"{model}__{lang}__{m}.npz" for m in ("bare", "averaged")}
        if not all(p.exists() for p in paths.values()):
            print(f"skip {system}: no averaged extraction")
            continue
        got = {}
        for mode, path in paths.items():
            reps = np.load(path, allow_pickle=False)["reps"].astype(np.float32)
            got[mode] = cosine_rdm(reps[:, layer, :])
        r = corr(tri_vec(sub_rdm(got["bare"], eval_idx)), tri_vec(sub_rdm(got["averaged"], eval_idx)))
        reliability[system] = r
        rdms[system] = got["bare"]
        print(f"{system:18s} layer {layer:2d}  bare-vs-averaged reliability = {r:+.3f}")

    if len(reliability) < 2:
        raise SystemExit(
            "Need at least two systems with both bare and averaged extractions. "
            "Run: python src/extract_representations.py --average-modes config"
        )

    rows = []
    for a, b in itertools.combinations(sorted(rdms), 2):
        rho = corr(tri_vec(sub_rdm(rdms[a], eval_idx)), tri_vec(sub_rdm(rdms[b], eval_idx)))
        denom = float(np.sqrt(max(reliability[a], 0.0) * max(reliability[b], 0.0)))
        normalised = rho / denom if denom > 1e-12 else np.nan
        rows.append({
            "system_a": a,
            "system_b": b,
            "heldout_spearman_rho": rho,
            "reliability_a": reliability[a],
            "reliability_b": reliability[b],
            "ceiling": denom,
            "normalised_agreement": normalised,
            # Two systems cannot agree with each other more than each agrees with
            # itself unless the reliability estimate is not measuring noise. When
            # this fires the normalisation is void for that pair, not impressive.
            "ceiling_violated": bool(np.isfinite(normalised) and normalised > 1.0),
            "both_systems_reliable": bool(min(reliability[a], reliability[b]) >= args.reliability_floor),
            "same_language": a.split("|")[1] == b.split("|")[1],
        })

    df = pd.DataFrame(rows)
    outpath = ROOT / args.out
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)

    unusable = sorted(s for s, r in reliability.items() if r < args.reliability_floor)
    if unusable:
        print(f"\nUNUSABLE (reliability < {args.reliability_floor}): {', '.join(unusable)}")
        print("  These systems' geometries do not survive a change of prompt. Agreement")
        print("  involving them is not interpretable and their normalisation is void.")
    violated = df[df["ceiling_violated"]]
    for _, r in violated.iterrows():
        print(f"CEILING VIOLATED: {r['system_a']} vs {r['system_b']} normalises to "
              f"{r['normalised_agreement']:.3f} > 1")

    print(f"\nmean reliability            {np.mean(list(reliability.values())):+.3f}")
    print(f"mean raw agreement          {df['heldout_spearman_rho'].mean():+.3f}")

    usable = df[df["both_systems_reliable"] & ~df["ceiling_violated"]]
    if not len(usable):
        print("\nNo pair has two reliable systems and a valid normalisation; "
              "the ceiling is not estimable from this extraction.")
        print(f"\nwrote {outpath}")
        return 0

    print(f"\nOver the {len(usable)} of {len(df)} pairs where both systems clear the floor "
          f"and the ceiling holds:")
    print(f"  mean agreement / ceiling  {usable['normalised_agreement'].mean():+.3f}")
    for same in (True, False):
        sub = usable[usable["same_language"] == same]
        if len(sub):
            label = "same-language" if same else "cross-language"
            print(f"  {label:14s} n={len(sub)}  raw {sub['heldout_spearman_rho'].mean():+.3f}"
                  f"  normalised {sub['normalised_agreement'].mean():+.3f}")
    print(f"\nwrote {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
