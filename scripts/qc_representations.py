#!/usr/bin/env python3
"""Quality-control the benchmark and any extracted representations.

Run this after extraction and before believing any analysis output. Everything
here was written because it caught a real defect on the first real run
(2026-09-03):

- bloom-560m in float16 on CUDA returned 80% NaN, and the extractor saved it
  without complaint. NaN RDMs then flow into `corr()`, which returns 0.0 for a
  non-finite result -- indistinguishable from "these systems do not agree".
- `learn` and `study` both carried the Chinese term 学习, so their Chinese
  representations were byte-identical in every model: a zero distance that means
  nothing about semantics, inside the matrix the whole comparison rests on.
- XGLM tokenises six Chinese terms to <unk>, collapsing 鲸 (whale) and 烹饪
  (cook) onto exactly the same vector, cosine 1.000000.

Usage:
    python scripts/qc_representations.py                 # benchmark + reps
    python scripts/qc_representations.py --tokenizers    # also probe vocab coverage
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_concepts():
    with (ROOT / "benchmark/concepts.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_benchmark(rows) -> list[str]:
    problems = []
    for lang in ("en", "zh"):
        dupes = {k: v for k, v in collections.Counter(r[lang].strip() for r in rows).items() if v > 1}
        if dupes:
            problems.append(f"{lang} surface form shared by several concepts: {dupes}")
    for r in rows:
        for field in ("concept_id", "en", "zh", "region"):
            if not r[field].strip():
                problems.append(f"empty {field} in {r}")
    print(f"benchmark: {len(rows)} concepts, "
          f"{'OK' if not problems else str(len(problems)) + ' problem(s)'}")
    return problems


def check_representations(rows) -> list[str]:
    problems = []
    files = sorted(glob.glob(str(ROOT / "outputs/representations/*.npz")))
    if not files:
        print("representations: none found (run src/extract_representations.py first)")
        return problems

    print(f"\n{'system':24s}{'shape':>18s}{'nonfinite':>11s}{'identical':>11s}"
          f"{'cos_mean':>10s}{'eff_rank':>10s}")
    for path in files:
        z = np.load(path, allow_pickle=False)
        reps = z["reps"].astype(np.float32)
        name = pathlib.Path(path).stem
        nonfinite = int((~np.isfinite(reps)).sum())
        last = reps[:, -1, :]
        identical = len(last) - len(np.unique(np.round(last, 5), axis=0))
        unit = last / np.maximum(np.linalg.norm(last, axis=1, keepdims=True), 1e-12)
        cos = unit @ unit.T
        np.fill_diagonal(cos, np.nan)
        centred = last - last.mean(0)
        sv = np.linalg.svd(centred, compute_uv=False)
        eff_rank = float((sv.sum() ** 2) / (sv ** 2).sum())
        print(f"{name:24s}{str(reps.shape):>18s}{nonfinite:>11d}{identical:>11d}"
              f"{np.nanmean(cos):>10.4f}{eff_rank:>10.1f}")

        if nonfinite:
            problems.append(
                f"{name}: {nonfinite} non-finite values. Re-extract with --dtype float32.")
        if identical:
            problems.append(f"{name}: {identical} concepts share an identical representation.")
        # Near-degenerate pairs are not automatically an error, but name them.
        iu = np.triu_indices(len(cos), 1)
        near = int(np.nansum(cos[iu] > 0.9999))
        if near:
            ids = [r["concept_id"] for r in rows]
            i, j = np.unravel_index(np.nanargmax(cos), cos.shape)
            print(f"    note: {near} pair(s) above cosine 0.9999, "
                  f"e.g. {ids[i]} ~ {ids[j]} at {cos[i, j]:.6f}")
        if eff_rank < 25:
            print(f"    note: effective rank {eff_rank:.1f} is very low; this space is "
                  "strongly anisotropic and its distances carry little structure")
    return problems


def check_tokenizers(rows) -> list[str]:
    from transformers import AutoTokenizer

    cfg = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    print(f"\n{'model':16s}{'lang':>5s}{'unk':>6s}{'pct':>8s}  concepts hitting <unk>")
    problems = []
    for key, name in cfg["models"].items():
        tok = AutoTokenizer.from_pretrained(name)
        if tok.unk_token_id is None:
            continue
        for lang in ("en", "zh"):
            hit = [r["concept_id"] for r in rows
                   if tok.unk_token_id in tok(r[lang])["input_ids"]]
            print(f"{key:16s}{lang:>5s}{len(hit):>6d}{len(hit) / len(rows) * 100:>7.1f}%  "
                  f"{', '.join(hit[:6])}")
            if hit:
                problems.append(
                    f"{key}/{lang}: {len(hit)} concepts tokenise to <unk> and are mutually "
                    f"indistinguishable in that system: {hit}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizers", action="store_true",
                    help="also probe vocabulary coverage (downloads tokenizers)")
    args = ap.parse_args()

    rows = load_concepts()
    problems = check_benchmark(rows) + check_representations(rows)
    if args.tokenizers:
        problems += check_tokenizers(rows)

    print()
    if problems:
        print(f"QC found {len(problems)} issue(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("QC clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
