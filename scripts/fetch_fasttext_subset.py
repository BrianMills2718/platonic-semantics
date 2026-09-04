#!/usr/bin/env python3
"""Pull only the benchmark's own words out of fastText's aligned vectors.

The English aligned file is 5.7 GB and the Chinese one 750 MB, and this project
needs 212 words from each. The files are ordered by corpus frequency, so the
whole thing streams through a filter and the connection closes as soon as every
wanted word has been seen. Downloads a few tens of MB instead of 6.4 GB.

Writes benchmark-scoped vectors to data/fasttext/<lang>.npz and reports coverage,
because a word that is missing is a real limit on the baseline, not a detail.
"""
from __future__ import annotations

import argparse
import csv
import io
import pathlib
import sys
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned"
OUT = ROOT / "data" / "fasttext"


def stream_subset(lang: str, wanted: set[str], max_lines: int) -> dict[str, np.ndarray]:
    url = f"{BASE}/wiki.{lang}.align.vec"
    found: dict[str, np.ndarray] = {}
    req = urllib.request.Request(url, headers={"User-Agent": "platonic-semantics/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        reader = io.TextIOWrapper(resp, encoding="utf-8", errors="replace")
        next(reader)  # header: "<n_words> <dim>"
        for n, line in enumerate(reader):
            if n >= max_lines or len(found) == len(wanted):
                break
            sp = line.find(" ")
            if sp < 0:
                continue
            word = line[:sp]
            if word in wanted and word not in found:
                found[word] = np.fromstring(line[sp + 1:], sep=" ", dtype=np.float32)
                if len(found) % 50 == 0:
                    print(f"  {lang}: {len(found)}/{len(wanted)} at line {n:,}", flush=True)
    print(f"  {lang}: scanned {n:,} lines, found {len(found)}/{len(wanted)}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-lines", type=int, default=600_000)
    args = ap.parse_args()

    with (ROOT / "benchmark/concepts.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    OUT.mkdir(parents=True, exist_ok=True)

    for lang in ("en", "zh"):
        terms = {r[lang]: r["concept_id"] for r in rows}
        print(f"{lang}: need {len(terms)} terms")
        found = stream_subset(lang, set(terms), args.max_lines)
        missing = sorted(set(terms) - set(found))
        dim = len(next(iter(found.values()))) if found else 0
        mat = np.zeros((len(rows), dim), dtype=np.float32)
        present = np.zeros(len(rows), dtype=bool)
        for i, r in enumerate(rows):
            v = found.get(r[lang])
            if v is not None and len(v) == dim:
                mat[i] = v
                present[i] = True
        np.savez_compressed(OUT / f"{lang}.npz", vectors=mat, present=present,
                            concept_ids=np.array([r["concept_id"] for r in rows]))
        print(f"  saved {OUT / f'{lang}.npz'}  dim={dim}  covered={int(present.sum())}/{len(rows)}")
        if missing:
            print(f"  MISSING ({len(missing)}): {', '.join(missing[:15])}"
                  + (" ..." if len(missing) > 15 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
