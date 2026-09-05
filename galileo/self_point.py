#!/usr/bin/env python3
"""The Self as an object in the space — Woelfel's most consequential move.

In Galileo the self is not special machinery. It is an object like any other,
judged against the same rod: "how far apart are *you* and war?" That is what
makes the theory operational, because distance from Self then predicts behaviour
rather than merely describing meaning. Woelfel found the brand nearer a person's
Self held the larger market share, and the medium nearer Self was used more.
Attitude change, in his framing, is the Self and an object moving closer.

Nothing in this repository's other routes can obtain it. Activations give a
geometry of concepts with no place in it for the model; word co-occurrence gives
whatever the text happened to mention. Only asking produces a Self-point, and it
costs exactly one more object.

Two things are measured here:

* **Where each model puts itself** among ten political concepts, in the same
  ratio-scale units as every other distance.
* **Whether the models agree about that.** They already agree strongly about the
  concepts (0.845). If they diverge about where *they* sit while agreeing about
  everything else, that divergence is about the models rather than about
  politics — and it is invisible to every other instrument.

The Self is elicited inside the same list as the other concepts, never in a
separate pass, because this instrument's reliability comes from judging many
pairs in one call (see `README.md`). Pulling the Self out would measure it with a
worse instrument than everything it is compared against.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import statistics
import time

import numpy as np

from elicit import elicit_averaged, pearson
from elicit_map import CONCEPTS, to_matrix, spearman

ROOT = pathlib.Path(__file__).resolve().parent

# Woelfel's respondents were asked about "Me". The second-person form is the
# equivalent when the respondent is being addressed rather than self-reporting.
SELF = "yourself"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--budget", type=float, default=2.00)
    ap.add_argument("--out", default="results/self_point.json")
    args = ap.parse_args()

    models = args.model or ["openrouter/openai/gpt-5.6-luna",
                            "openrouter/z-ai/glm-5.2",
                            "openrouter/deepseek/deepseek-v4-flash"]
    concepts = CONCEPTS + [SELF]
    pairs = list(itertools.combinations(concepts, 2))
    rod = ("good", "evil", 100)
    trace = f"galileo-self-{int(time.time())}"
    iu = np.triu_indices(len(concepts), 1)
    si = concepts.index(SELF)

    print(f"{len(concepts)} objects including the Self, {len(pairs)} pairs, "
          f"{args.permutations} orders x 2 estimates\n")

    D, rel = {}, {}
    for m in models:
        rng = random.Random(20260905)
        short = m.split("/")[-1]
        e1, _, _ = elicit_averaged(m, pairs, rod, trace, args.budget,
                                   args.permutations, rng, rescale=True)
        e2, _, _ = elicit_averaged(m, pairs, rod, trace, args.budget,
                                   args.permutations, rng, rescale=True)
        rel[short] = pearson(e1, e2)
        D[short] = to_matrix(concepts, pairs, [statistics.fmean(p) for p in zip(e1, e2)])
        print(f"  {short:<22s} agrees with itself  {rel[short]:.3f}")

    names = list(D)
    print(f"\nDISTANCE FROM THE SELF (smaller = the model places itself nearer):\n")
    w = max(len(c) for c in CONCEPTS) + 2
    print(" " * w + "".join(f"{n[:11]:>13s}" for n in names) + f"{'mean':>9s}")
    rows = {}
    for i, c in enumerate(CONCEPTS):
        vals = [D[n][si, i] for n in names]
        rows[c] = vals
        print(f"{c:<{w}s}" + "".join(f"{v:>13.2f}" for v in vals)
              + f"{statistics.fmean(vals):>9.2f}")

    order = sorted(CONCEPTS, key=lambda c: statistics.fmean(rows[c]))
    print(f"\n  nearest the Self: {', '.join(order[:3])}")
    print(f"  furthest:         {', '.join(order[-3:])}")

    # Do the models agree about where they sit, given they agree about the concepts?
    print(f"\nagreement between models:")
    print(f"{'':22s}{'about the concepts':>20s}{'about the Self':>17s}")
    for a, b in itertools.combinations(names, 2):
        keep = [i for i in range(len(concepts)) if i != si]
        sub = np.ix_(keep, keep)
        ju = np.triu_indices(len(keep), 1)
        conc = spearman(D[a][sub][ju], D[b][sub][ju])
        selfv = spearman(np.array([D[a][si, i] for i in keep], float),
                         np.array([D[b][si, i] for i in keep], float))
        print(f"  {a[:10]:<10s} vs {b[:10]:<10s}{conc:>18.3f}{selfv:>17.3f}")

    conc_all = statistics.fmean([spearman(
        D[a][np.ix_([i for i in range(len(concepts)) if i != si],
                    [i for i in range(len(concepts)) if i != si])][
            np.triu_indices(len(concepts) - 1, 1)],
        D[b][np.ix_([i for i in range(len(concepts)) if i != si],
                    [i for i in range(len(concepts)) if i != si])][
            np.triu_indices(len(concepts) - 1, 1)])
        for a, b in itertools.combinations(names, 2)])
    self_all = statistics.fmean([spearman(
        np.array([D[a][si, i] for i in range(len(concepts)) if i != si], float),
        np.array([D[b][si, i] for i in range(len(concepts)) if i != si], float))
        for a, b in itertools.combinations(names, 2)])
    print(f"\n  mean about the concepts {conc_all:+.3f}")
    print(f"  mean about the Self     {self_all:+.3f}")
    if self_all < conc_all - 0.15:
        print("  -> the models agree about politics but NOT about where they stand in it")
    elif self_all > conc_all + 0.15:
        print("  -> the models agree about themselves more than about politics")
    else:
        print("  -> no difference the instrument can resolve")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "concepts": concepts, "self": SELF, "rod": list(rod),
        "permutations": args.permutations, "models": names,
        "self_agreement": rel,
        "distance_from_self": {c: {n: float(D[n][si, i]) for n in names}
                               for i, c in enumerate(CONCEPTS)},
        "agreement_about_concepts": conc_all, "agreement_about_self": self_all,
        "distances": {k: v.tolist() for k, v in D.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
