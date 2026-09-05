#!/usr/bin/env python3
"""Is the nesting real, or an artefact of two models sharing one batch partition?

`nesting_elicited.py` found the elicited spaces recover ~66% toward perfect
nesting, against 6-23% for activations in run 001. That would be strong support
for the lattice account in `docs/THEORY.md`. But every space in that comparison
was assembled through the *same* batch partition and the same anchor pairs,
because the shuffle was seeded identically. Equating is imperfect -- anchor
spread runs 0.17-0.20 -- so the same block structure is imposed on both matrices,
and clustering could be recovering shared batch artefacts rather than shared
semantics.

The control is one model re-elicited with a different batch partition. Three
comparisons then separate the explanations, and they disagree only if the
confound is real:

    A. model X vs model Y, SAME partition      <- the original number
    B. model X vs model Y, DIFFERENT partition <- semantics only, no shared blocks
    C. model X vs ITSELF, different partition  <- how much the pipeline alone
                                                  reproduces, an upper bound on
                                                  what any pair could show

If B holds near A, the batch structure was not doing the work and the nesting is
semantic. If B collapses toward chance, it was, and the headline is withdrawn.
C bounds both: no cross-model comparison can exceed what the same model measured
twice through different partitions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from nesting_by_scale import unexplained  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent


def load(p):
    j = json.loads(pathlib.Path(p).read_text())
    return j["model"].split("/")[-1], j.get("batch_seed"), np.array(j["distances"]), j["concepts"]


def recovery(A, B, ks, rng, nulls=300):
    out = []
    for k in ks:
        pa = fcluster(linkage(squareform(A, checks=False), method="average"), k, "maxclust")
        pb = fcluster(linkage(squareform(B, checks=False), method="average"), k, "maxclust")
        obs = unexplained(pa, pb)
        null = []
        for _ in range(nulls):
            x, y = list(pa), list(pb)
            rng.shuffle(x); rng.shuffle(y)
            null.append(unexplained(np.array(x), np.array(y)))
        mu = float(np.mean(null))
        out.append((mu - obs) / mu if mu > 0 else 0.0)
    return float(np.mean(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--same", nargs="+", default=["results/scaled_gpt-5.6-luna.json",
                                                  "results/scaled_deepseek-v4-flash.json",
                                                  "results/scaled_glm-5.2.json"])
    ap.add_argument("--control", default="results/control_luna_seed777.json")
    ap.add_argument("--ks", nargs="*", type=int, default=[4, 6, 8, 10, 12, 15])
    ap.add_argument("--out", default="results/nesting_control.json")
    args = ap.parse_args()

    spaces, concepts = {}, None
    for p in args.same:
        f = ROOT / p
        if not f.exists():
            print(f"  (missing {p}, skipping)"); continue
        name, seed, M, c = load(f)
        concepts = concepts or c
        spaces[name] = (seed, M)
    cname, cseed, cM, _ = load(ROOT / args.control)

    rng = random.Random(11)
    print(f"{len(concepts)} concepts. Control: {cname} re-elicited on batch seed "
          f"{cseed} instead of {list(spaces.values())[0][0]}.\n")

    print("A. different models, SAME batch partition")
    a = []
    for x, y in itertools.combinations(spaces, 2):
        r = recovery(spaces[x][1], spaces[y][1], args.ks, rng)
        a.append(r); print(f"     {x:<20s} vs {y:<20s} {r*100:>6.1f}%")

    print("\nB. different models, DIFFERENT batch partitions  <- the decisive one")
    b = []
    for x in spaces:
        if x == cname:
            continue
        r = recovery(cM, spaces[x][1], args.ks, rng)
        b.append(r); print(f"     {cname}(seed {cseed}) vs {x:<20s} {r*100:>6.1f}%")

    print("\nC. same model, different partitions — the pipeline's own ceiling")
    c = recovery(cM, spaces[cname][1], args.ks, rng)
    print(f"     {cname} vs itself {c*100:>6.1f}%")

    ma, mb = float(np.mean(a)), float(np.mean(b))
    print(f"\n  A same-partition   {ma*100:>5.1f}%")
    print(f"  B cross-partition  {mb*100:>5.1f}%")
    print(f"  C ceiling          {c*100:>5.1f}%")
    print(f"  activations (run 001, 212 concepts): 5.9-22.9%")

    drop = ma - mb
    if drop > 0.20:
        verdict = ("ARTEFACT: most of the nesting came from the shared batch "
                   "partition, not from shared semantics. The headline is withdrawn.")
    elif mb < 0.30:
        verdict = ("the cross-partition figure is low in absolute terms; the "
                   "elicited spaces do not nest much more than the activation ones")
    else:
        verdict = ("the nesting SURVIVES a different batch partition, so it is "
                   "semantic rather than an artefact of how pairs were grouped")
    print(f"\n  VERDICT: {verdict}")

    (ROOT / args.out).write_text(json.dumps({
        "concepts": concepts, "control_model": cname, "control_seed": cseed,
        "same_partition": ma, "cross_partition": mb, "self_ceiling": c,
        "drop": drop, "verdict": verdict}, indent=1), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
