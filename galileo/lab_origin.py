#!/usr/bin/env python3
"""Does model agreement track the LAB that built it, on politically loaded concepts?

Every convergence result in this project carries one standing deflationary
explanation: models trained on overlapping corpora may agree because their
*sources* overlap, not because they found a shared structure. Nothing measured so
far bears on it, because every concept set used has been culturally neutral —
animals, then astronomy. Neutral concepts cannot distinguish the two accounts,
since a platonic structure and a shared corpus both predict agreement about
`dog`.

Culturally loaded concepts can. If the two Chinese-lab models (`glm-5.2`,
`deepseek-v4-flash`) agree with each other more than either agrees with the US
model (`gpt-5.6-luna`) **on loaded concepts specifically, while matching on
neutral ones**, then training provenance is doing the work and the convergence is
culturally contingent rather than structural.

The design has one load-bearing property: **loaded and neutral concepts are
elicited in the SAME run**, over the same 40-concept set, against the same rod,
in the same batches. Running them separately would confound the contrast with
everything else that differs between two runs — the failure recorded as
`lrn-20260826T231358274274Z-fc1b2d26d4`, where a comparison's two arms were not
granted identical conditions and the first result was an artefact of that.

So the statistic is an **interaction**, not a difference:

    (same-lab agreement − cross-lab agreement) on LOADED pairs
  − (same-lab agreement − cross-lab agreement) on NEUTRAL pairs

Zero means lab origin does not matter more for politics than for rocks. Clearly
positive means it does, and the deflationary explanation gains real support.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import statistics

import numpy as np
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parent

# Provenance, not capability. The claim under test is about where a model was
# built, so this is the grouping the interaction is computed over.
LAB = {"gpt-5.6-luna": "US", "glm-5.2": "CN", "deepseek-v4-flash": "CN"}


def spearman(a, b):
    # Average ranks for ties. argsort(argsort(x)) hands tied values arbitrary
    # distinct ranks, which biases rho and makes it depend on input order.
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="loaded_*.json")
    ap.add_argument("--concepts", default="concepts_loaded.csv")
    ap.add_argument("--out", default="results/lab_origin.json")
    args = ap.parse_args()

    kind = {r[1]: r[3] for r in list(csv.reader(open(ROOT / args.concepts)))[1:]}
    spaces, concepts = {}, None
    for f in sorted((ROOT / "results").glob(args.glob)):
        j = json.loads(f.read_text())
        concepts = concepts or j["concepts"]
        spaces[j["model"].split("/")[-1]] = np.array(j["distances"])
    if len(spaces) < 3:
        raise RuntimeError(f"need all three models, have {sorted(spaces)}")

    n = len(concepts)
    loaded = [i for i, c in enumerate(concepts) if kind.get(c) == "loaded"]
    neutral = [i for i, c in enumerate(concepts) if kind.get(c) == "neutral"]
    print(f"{n} concepts: {len(loaded)} loaded, {len(neutral)} neutral")
    print(f"labs: " + ", ".join(f"{m}={LAB.get(m,'?')}" for m in spaces) + "\n")

    def agree(a, b, idx):
        sub = np.ix_(idx, idx)
        ju = np.triu_indices(len(idx), 1)
        return spearman(spaces[a][sub][ju], spaces[b][sub][ju])

    out = {}
    for label, idx in (("loaded", loaded), ("neutral", neutral)):
        same, cross = [], []
        print(f"{label} concepts ({len(idx)*(len(idx)-1)//2} pairs):")
        for a, b in itertools.combinations(spaces, 2):
            v = agree(a, b, idx)
            tag = "same lab" if LAB[a] == LAB[b] else "cross lab"
            (same if LAB[a] == LAB[b] else cross).append(v)
            print(f"  {a:<20s} vs {b:<20s} {v:+.3f}   {tag}")
        gap = statistics.fmean(same) - statistics.fmean(cross)
        out[label] = {"same_lab": statistics.fmean(same),
                      "cross_lab": statistics.fmean(cross), "gap": gap}
        print(f"  same-lab {statistics.fmean(same):+.3f}   "
              f"cross-lab {statistics.fmean(cross):+.3f}   gap {gap:+.3f}\n")

    inter = out["loaded"]["gap"] - out["neutral"]["gap"]
    print(f"  INTERACTION (loaded gap - neutral gap): {inter:+.3f}")
    if inter > 0.10:
        verdict = ("lab origin matters MORE for politically loaded concepts than for "
                   "neutral ones -- support for the training-provenance explanation")
    elif inter < -0.10:
        verdict = ("lab origin matters LESS for loaded concepts, which is the opposite "
                   "of what shared-corpus provenance predicts")
    else:
        verdict = ("no interaction the instrument can resolve: same-lab models are not "
                   "specially alike on politics")
    print(f"  VERDICT: {verdict}")

    (ROOT / args.out).write_text(json.dumps(
        {"concepts": concepts, "labs": LAB, "by_kind": out,
         "interaction": inter, "verdict": verdict}, indent=1), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
