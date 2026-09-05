#!/usr/bin/env python3
"""Ask a model for Galileo-style magnitude estimates, and check they are ratio-scale.

Woelfel's method gets its power from one property: distances are reported as
ratios to a declared reference pair, so two respondents' numbers are in the same
units without any alignment. That is what cosine distances from two different
models are not, and it is the reason this route exists alongside the
activation-based one.

The property is an assumption about the respondent. For humans it was validated
across decades of magnitude-estimation psychophysics. For a language model it is
unverified, and everything downstream is void if it fails, so `--check-stability`
runs before any map gets built:

  order         how much a pair's answer moves with its position in the list
  test-retest   two independent averaged runs -> should agree
  rod-swap      a different reference pair -> should RESCALE, not scramble

The first pilot failed the order check outright (r=0.53 between a list and its
reverse), with the reversed run visibly coarser and clustering on multiples of
five -- a respondent filling in a list rather than estimating a magnitude. This
is a known hazard in human magnitude estimation too, and the standard remedy is
the one used here: randomise the presentation order on every call and average
over several permutations. Order sensitivity is still measured rather than
assumed away, as the spread across permutations for each pair.

The rod-swap is the one that actually tests ratio-scaling. If "good/evil = 100"
and "hot/cold = 50" produce distance sets related by a single constant, the
judgments carry ratio information. If the correlation is high but the ratio
wanders, they are interval or ordinal and the method does not apply.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, "/home/brian/code/llm_client")

from llm_client import call_llm_json_schema  # noqa: E402
from llm_client.prompts import render_prompt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
PROMPT = ROOT / "prompts" / "magnitude_estimation.yaml"

# Small, deliberately heterogeneous: concrete objects, animals, institutions,
# emotions and abstractions. A set drawn from one domain would make every
# distance similar and hide exactly the variation the method needs to resolve.
PILOT_CONCEPTS = [
    "dog", "wolf", "hammer", "hospital",
    "anger", "justice", "river", "democracy",
]


def schema_for(n: int) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["distances"],
        "properties": {
            "distances": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "description": (
                    "One distance per numbered pair, in the order given. Each is a "
                    "positive number on the same scale as the stated reference "
                    "distance. Not a similarity score and not bounded by 1."
                ),
                "items": {"type": "number"},
            }
        },
    }


# Magnitude estimation is a psychophysical judgment, not a reasoning task:
# Woelfel's respondents answer immediately against the rod. Letting the model
# deliberate would measure its reasoning about semantics rather than its
# semantics, so reasoning is explicitly off. The policy forbids inheriting a
# provider default, which is the right call -- it would have been an unnoticed
# confound varying by model.
REASONING = "none"


def elicit_isolated(model, pairs, rod, trace_id, budget, relation=None):
    """One pair per call: the only way to remove list effects rather than average them.

    Permutation averaging failed to rescue the batched form (test-retest fell
    from 0.900 to 0.827 when averaging five orders), which rules out independent
    per-pair noise -- averaging would have helped. The error is therefore
    list-level: a whole call shifts together, so every estimate inside it shares
    the same displacement and no amount of reordering averages it away. Asking
    for one judgment at a time is the only elicitation that has no list.
    """
    out = []
    for pair in pairs:
        vals, _ = elicit(model, [pair], rod, trace_id, budget, relation)
        out.append(vals[0])
    return out


def elicit_averaged(model, pairs, rod, trace_id, budget, n_perms, rng, relation=None):
    """Average over randomised presentation orders, returning (mean, per_perm).

    Each call sees the pairs in a fresh random order and the answers are mapped
    back to canonical order before averaging, so list position cannot correlate
    with any particular pair across the set.
    """
    per_perm = []
    for _ in range(n_perms):
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        shuffled = [pairs[i] for i in idx]
        vals, _ = elicit(model, shuffled, rod, trace_id, budget, relation)
        restored = [0.0] * len(pairs)
        for slot, original_i in enumerate(idx):
            restored[original_i] = vals[slot]
        per_perm.append(restored)
    mean = [statistics.fmean(col) for col in zip(*per_perm)]
    return mean, per_perm


def elicit(model, pairs, rod, trace_id, budget, relation=None):
    messages = render_prompt(
        PROMPT,
        rod_a=rod[0], rod_b=rod[1], rod_value=rod[2],
        rod_value_doubled=rod[2] * 2, relation=relation, pairs=pairs,
    )
    data, result = call_llm_json_schema(
        model, messages, schema_for(len(pairs)),
        schema_name="galileo_distances",
        task="galileo-magnitude-estimation",
        trace_id=trace_id,
        max_budget=budget,
        reasoning_effort=REASONING,
        **({} if model == "openrouter/openai/gpt-5.6-luna" else {
            "model_justification":
                "Cross-model comparison is the measurement: this run tests whether "
                "unstable magnitude judgments are a property of one model or of the "
                "elicitation method, which cannot be answered from the default model alone."
        }),
    )
    vals = [float(v) for v in data["distances"]]
    if len(vals) != len(pairs):
        raise RuntimeError(f"asked for {len(pairs)} distances, got {len(vals)}")
    if any(v < 0 for v in vals):
        raise RuntimeError("negative distance returned; the scale is not being honoured")
    return vals, result


def pearson(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openrouter/openai/gpt-5.6-luna")
    ap.add_argument("--check-stability", action="store_true",
                    help="run the three validity checks and write a report; build nothing")
    ap.add_argument("--isolated", action="store_true",
                    help="one pair per call: removes list effects instead of averaging them")
    ap.add_argument("--permutations", type=int, default=5,
                    help="random presentation orders averaged per estimate")
    ap.add_argument("--budget", type=float, default=2.00)
    ap.add_argument("--out", default="results/stability.json")
    args = ap.parse_args()

    concepts = PILOT_CONCEPTS
    pairs = list(itertools.combinations(concepts, 2))
    trace = f"galileo-stability-{int(time.time())}"
    rod_a = ("good", "evil", 100)
    rod_b = ("hot", "cold", 50)
    rng = random.Random(20260905)
    P = args.permutations

    print(f"model   {args.model}")
    print(f"pilot   {len(concepts)} concepts, {len(pairs)} pairs, {P} random orders per estimate")
    print(f"rods    {rod_a[0]}/{rod_a[1]}={rod_a[2]}   {rod_b[0]}/{rod_b[1]}={rod_b[2]}\n")

    if args.isolated:
        print("  elicitation: ONE PAIR PER CALL (no list)\n")
        a1 = elicit_isolated(args.model, pairs, rod_a, trace, args.budget)
        print(f"  estimate 1 (rod A)  {[round(v) for v in a1]}")
        a2 = elicit_isolated(args.model, pairs, rod_a, trace, args.budget)
        print(f"  estimate 2 (rod A)  {[round(v) for v in a2]}")
        b1 = elicit_isolated(args.model, pairs, rod_b, trace, args.budget)
        print(f"  estimate 3 (rod B)  {[round(v) for v in b1]}")
        a1_perms = [a1]
    else:
        a1, a1_perms = elicit_averaged(args.model, pairs, rod_a, trace, args.budget, P, rng)
        print(f"  estimate 1 (rod A)  {[round(v) for v in a1]}")
        a2, _ = elicit_averaged(args.model, pairs, rod_a, trace, args.budget, P, rng)
        print(f"  estimate 2 (rod A)  {[round(v) for v in a2]}")
        b1, _ = elicit_averaged(args.model, pairs, rod_b, trace, args.budget, P, rng)
        print(f"  estimate 3 (rod B)  {[round(v) for v in b1]}")

    # How much does presentation order move a single pair's answer? Measured on
    # the raw per-permutation values, not on the averages that hide it.
    order_cv = statistics.fmean([
        statistics.pstdev(col) / statistics.fmean(col)
        for col in zip(*a1_perms) if statistics.fmean(col) > 0
    ]) if len(a1_perms) > 1 else float("nan")
    retest = pearson(a1, a2)
    rodcorr = pearson(a1, b1)
    ratios = [b / a for a, b in zip(a1, b1) if a > 0]
    cv = statistics.pstdev(ratios) / statistics.fmean(ratios) if ratios else float("nan")
    expected = rod_b[2] / rod_a[2]

    print(f"\n  order sensitivity  CV {order_cv:.3f}   (spread across orders for one pair)")
    print(f"  test-retest r      {retest:.3f}   (want > 0.9)")
    print(f"  rod-swap r         {rodcorr:.3f}   (want > 0.9)")
    print(f"  rod-swap ratio     {statistics.fmean(ratios):.3f} +/- CV {cv:.3f}"
          f"   (want CV < 0.2; naive rescale would give {expected:.2f})")

    ok = retest > 0.9 and rodcorr > 0.9 and cv < 0.2
    print(f"\n  VERDICT: {'ratio-scale judgments look usable' if ok else 'NOT ratio-stable -- do not build maps on this'}")
    runs = {"a1": a1, "a2": a2, "b1": b1, "a1_perms": a1_perms}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "concepts": concepts, "pairs": [list(p) for p in pairs],
        "rod_a": list(rod_a), "rod_b": list(rod_b), "runs": runs,
        "permutations": P, "order_sensitivity_cv": order_cv,
        "test_retest_r": retest, "rod_swap_r": rodcorr,
        "rod_swap_ratio_mean": statistics.fmean(ratios) if ratios else None,
        "rod_swap_ratio_cv": cv, "passes": ok,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
