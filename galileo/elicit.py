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
import time

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
    """Each judgment carries the number of the pair it belongs to.

    A bare list is mapped by position, so one extra or missing value silently
    shifts every judgment onto the wrong pair -- and with a whole run of these
    being averaged, that corruption would not announce itself. It did announce
    itself here only because the array length was pinned: a model returned 46
    distances for 45 pairs and the run died. Labelling each judgment makes the
    failure detectable rather than merely fatal, and lets a stray extra entry be
    dropped instead of discarding the call.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["distances"],
        "properties": {
            "distances": {
                "type": "array",
                # Deliberately 1, not n. Completeness is enforced by the caller,
                # which drops a thin permutation and fails only when a pair is
                # missing from EVERY order. Leaving minItems at n meant a model
                # returning an empty or short list blew up inside validation
                # before that logic ran, killing a 20-minute multi-batch run --
                # two different failure modes fighting each other over the same
                # rule.
                "minItems": 1,
                "description": (
                    "One entry per numbered pair. Each is a positive number on the "
                    "same scale as the stated reference distance -- not a similarity "
                    "score and not bounded by 1."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pair", "distance"],
                    "properties": {
                        "pair": {"type": "integer",
                                 "description": "The number of the pair, as listed."},
                        "distance": {"type": "number"},
                    },
                },
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


def elicit_contextual(model, pairs, rod, trace_id, budget, concepts, relation=None):
    """One judgment per call, but with the whole concept set shown as context.

    The two prior failures point here. Batched elicitation is anchored but
    carries a list-level bias; one-pair-per-call has no list bias but no anchor
    either, and collapses. Those two effects were confounded. Showing the concept
    set without asking for a list separates them: the model can calibrate its
    range against the full study, and only one number is requested, so no answer
    has a position among other answers.
    """
    return [elicit(model, [pair], rod, trace_id, budget, relation, concepts)[0][0]
            for pair in pairs]


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


def rescale_to_unit_mean(vals):
    """Divide a call's answers by their own mean, cancelling a whole-call shift.

    The diagnosis was that error here is list-level: a whole call moves together,
    so averaging permutations cannot remove it (and empirically made things
    worse). If that shift is multiplicative, dividing each call by its own mean
    removes it exactly while leaving every ratio inside the call untouched --
    and ratios are the only thing Woelfel's method claims to measure.

    Checked against the pilot data before being used: order sensitivity falls
    from 0.475 to 0.195 for GLM-5.2 and 0.236 to 0.189 for Luna. So most of the
    instability really was one number per call, not noise in each judgment.
    """
    m = statistics.fmean([v for v in vals if v is not None and v > 0]) or 1.0
    return [None if v is None else v / m for v in vals]


def elicit_averaged(model, pairs, rod, trace_id, budget, n_perms, rng,
                    relation=None, rescale=True):
    """Average over randomised presentation orders, returning (mean, per_perm).

    Each call sees the pairs in a fresh random order and the answers are mapped
    back to canonical order before averaging, so list position cannot correlate
    with any particular pair across the set. With `rescale`, each call is first
    divided by its own mean, which is what makes the averaging work at all.

    A call that omits some judgments is kept with gaps (None) rather than
    killing the run, per the completeness policy in `schema_for`: a permutation
    answering fewer than MIN_PERM_COVERAGE of the pairs is too thin to rescale
    and is dropped, and the run fails only when a pair is missing from EVERY
    kept permutation. Each pair's mean is over the permutations that saw it.
    """
    per_perm, raw_perm = [], []
    for _ in range(n_perms):
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        shuffled = [pairs[i] for i in idx]
        vals, _ = elicit(model, shuffled, rod, trace_id, budget, relation,
                         allow_partial=True)
        restored = [None] * len(pairs)
        for slot, original_i in enumerate(idx):
            restored[original_i] = vals[slot]
        if sum(v is not None for v in restored) < len(pairs) * MIN_PERM_COVERAGE:
            continue                      # too thin to rescale; drop this order
        raw_perm.append(restored)
        per_perm.append(rescale_to_unit_mean(restored) if rescale else restored)
    if not per_perm:
        raise RuntimeError(f"no usable permutation out of {n_perms}")
    mean, raw_mean = _mean_seen(per_perm, pairs), _mean_seen(raw_perm, pairs)
    return mean, per_perm, raw_mean


# Same threshold `scale_elicit` uses for the same decision.
MIN_PERM_COVERAGE = 0.7


def _mean_seen(perms, pairs):
    out = []
    for pair, col in zip(pairs, zip(*perms)):
        seen = [v for v in col if v is not None]
        if not seen:
            raise RuntimeError(f"pair {pair} was missing from every permutation")
        out.append(statistics.fmean(seen))
    return out


def elicit(model, pairs, rod, trace_id, budget, relation=None, context_concepts=None,
           allow_partial=False):
    messages = render_prompt(
        PROMPT,
        rod_a=rod[0], rod_b=rod[1], rod_value=rod[2],
        rod_value_doubled=rod[2] * 2, relation=relation, pairs=pairs,
        context_concepts=context_concepts,
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
    by_pair = {}
    for entry in data["distances"]:
        i = int(entry["pair"])
        if not 1 <= i <= len(pairs):
            continue                      # a stray entry outside the list; drop it
        by_pair[i] = float(entry["distance"])
    missing = [i for i in range(1, len(pairs) + 1) if i not in by_pair]
    if missing and not allow_partial:
        raise RuntimeError(
            f"no judgment returned for pair(s) {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} of {len(pairs)}")
    # With `allow_partial` a dropped judgment becomes a gap rather than a dead
    # run. Some models silently omit a few entries from a long list, and killing
    # a whole multi-batch elicitation over five missing judgments out of 53
    # throws away work that repeated permutations can recover. The caller is then
    # responsible for failing if a pair is missing from EVERY permutation, which
    # is the level at which the data really is absent.
    vals = [by_pair.get(i) for i in range(1, len(pairs) + 1)]
    if any(v is not None and v < 0 for v in vals):
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
    ap.add_argument("--contextual", action="store_true",
                    help="one pair per call WITH the concept set shown; separates anchoring from list position")
    ap.add_argument("--isolated", action="store_true",
                    help="one pair per call: removes list effects instead of averaging them")
    ap.add_argument("--permutations", type=int, default=5,
                    help="random presentation orders averaged per estimate")
    ap.add_argument("--no-rescale", action="store_true",
                    help="skip per-call rescaling (reproduces the original failing runs)")
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

    if args.contextual:
        print("  elicitation: ONE PAIR PER CALL, concept set shown as context\n")
        a1 = elicit_contextual(args.model, pairs, rod_a, trace, args.budget, concepts)
        print(f"  estimate 1 (rod A)  {[round(v) for v in a1]}")
        a2 = elicit_contextual(args.model, pairs, rod_a, trace, args.budget, concepts)
        print(f"  estimate 2 (rod A)  {[round(v) for v in a2]}")
        b1 = elicit_contextual(args.model, pairs, rod_b, trace, args.budget, concepts)
        print(f"  estimate 3 (rod B)  {[round(v) for v in b1]}")
        a1_perms = [a1]
    elif args.isolated:
        print("  elicitation: ONE PAIR PER CALL (no list)\n")
        a1 = elicit_isolated(args.model, pairs, rod_a, trace, args.budget)
        print(f"  estimate 1 (rod A)  {[round(v) for v in a1]}")
        a2 = elicit_isolated(args.model, pairs, rod_a, trace, args.budget)
        print(f"  estimate 2 (rod A)  {[round(v) for v in a2]}")
        b1 = elicit_isolated(args.model, pairs, rod_b, trace, args.budget)
        print(f"  estimate 3 (rod B)  {[round(v) for v in b1]}")
        a1_perms = [a1]
    else:
        rs = not args.no_rescale
        print(f"  per-call rescaling: {'ON' if rs else 'OFF'}\n")
        a1, a1_perms, a1_raw = elicit_averaged(args.model, pairs, rod_a, trace,
                                               args.budget, P, rng, rescale=rs)
        print(f"  estimate 1 (rod A)  {[round(v, 2) for v in a1]}")
        a2, a2_perms, _ = elicit_averaged(args.model, pairs, rod_a, trace,
                                          args.budget, P, rng, rescale=rs)
        print(f"  estimate 2 (rod A)  {[round(v, 2) for v in a2]}")
        b1, b1_perms, b1_raw = elicit_averaged(args.model, pairs, rod_b, trace,
                                               args.budget, P, rng, rescale=rs)
        print(f"  estimate 3 (rod B)  {[round(v, 2) for v in b1]}")
        # Rescaling normalises absolute magnitude away, so the rod-ratio test --
        # which asks whether a different rod rescales everything by ONE constant --
        # has to be run on the raw numbers or it is true by construction.
        runs_extra = {"a2_perms": a2_perms, "b1_perms": b1_perms,
                      "a1_raw": a1_raw, "b1_raw": b1_raw}

    # How much does presentation order move a single pair's answer? Measured on
    # the raw per-permutation values, not on the averages that hide it.
    seen_cols = [[v for v in col if v is not None] for col in zip(*a1_perms)]
    order_cv = statistics.fmean([
        statistics.pstdev(col) / statistics.fmean(col)
        for col in seen_cols if len(col) > 1 and statistics.fmean(col) > 0
    ]) if len(a1_perms) > 1 else float("nan")
    retest = pearson(a1, a2)
    rodcorr = pearson(a1, b1)
    ra = locals().get("a1_raw") or a1
    rb = locals().get("b1_raw") or b1
    ratios = [y / x for x, y in zip(ra, rb) if x > 0]
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
    runs.update(locals().get("runs_extra", {}))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "concepts": concepts, "pairs": [list(p) for p in pairs],
        "rod_a": list(rod_a), "rod_b": list(rod_b), "runs": runs,
        "permutations": P, "order_sensitivity_cv": order_cv,
        "rescaled": not args.no_rescale,
        "test_retest_r": retest, "rod_swap_r": rodcorr,
        "rod_swap_ratio_mean": statistics.fmean(ratios) if ratios else None,
        "rod_swap_ratio_cv": cv, "passes": ok,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
