"""The check that the pilot did not have, and that its first real run needed.

Feed the analysis pure random geometry -- no semantics of any kind -- and the
relation tests must report nothing. The original null resampled each pair's
target independently, which destroyed the target reuse present in the observed
probes and made the null easier than the data. On random geometry it flagged
IsA (27 pairs over 9 distinct targets) at p <= .05 in 12/12 runs, 11/12 of them
surviving BH-FDR, at a rate monotone in the target-reuse ratio.

These tests fail if that ever comes back.
"""
from __future__ import annotations

import collections
import csv
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze_semantic_geometry as A  # noqa: E402

SEEDS = 8
PERMUTATIONS = 100
DIMS = 64
# Nominal alpha is .05. Over 8 seeds a correctly calibrated test is expected to
# fire 0-1 times; 3+ is a real calibration failure, not sampling noise.
MAX_FALSE_POSITIVES = 2


def _load():
    rows = list(csv.DictReader((ROOT / "benchmark" / "concepts.csv").open(encoding="utf-8")))
    index = {r["concept_id"]: i for i, r in enumerate(rows)}
    rel = collections.defaultdict(list)
    for r in csv.DictReader((ROOT / "benchmark" / "relations.csv").open(encoding="utf-8")):
        rel[r["relation"]].append((index[r["source"]], index[r["target"]]))
    return rows, [r["region"] for r in rows], dict(rel)


CONCEPTS, REGIONS, RELATIONS = _load()


def test_every_relation_has_distinct_targets():
    """Benchmark design constraint. Reuse makes the relation tests untestable:
    the correct multiplicity-preserving null converges on the observed data."""
    for rel, pairs in RELATIONS.items():
        ratio = A.target_reuse_ratio(pairs)
        assert ratio == 1.0, f"{rel} reuses targets (ratio {ratio:.2f})"


def test_no_pair_appears_under_two_relations():
    seen = collections.Counter(p for pairs in RELATIONS.values() for p in pairs)
    assert not [k for k, c in seen.items() if c > 1]


def test_random_target_null_preserves_target_multiplicity():
    rng = np.random.default_rng(0)
    n = len(CONCEPTS)
    # Deliberately reusing pairs, which is what the old null mishandled.
    pairs = [(0, 5), (1, 5), (2, 5), (3, 9), (4, 9), (6, 11)]
    for region_matched in (False, True):
        for _ in range(50):
            out = A.random_target_pairs(
                pairs, np.arange(n), rng,
                concept_regions=REGIONS, region_matched=region_matched,
            )
            assert [a for a, _ in out] == [a for a, _ in pairs]
            before = sorted(collections.Counter(b for _, b in pairs).values())
            after = sorted(collections.Counter(b for _, b in out).values())
            assert after == before, f"multiplicity {before} -> {after}"
            assert all(a != b for a, b in out)


def test_permuted_pairing_preserves_both_multisets():
    rng = np.random.default_rng(0)
    pairs = RELATIONS["IsA"]
    for _ in range(50):
        out = A.permuted_pairing_pairs(pairs, rng)
        assert sorted(a for a, _ in out) == sorted(a for a, _ in pairs)
        assert sorted(b for _, b in out) == sorted(b for _, b in pairs)
        assert all(a != b for a, b in out)


@pytest.mark.parametrize("null_kind", ["region_matched", "permuted_pairing"])
def test_relation_test_reports_nothing_on_random_geometry(null_kind):
    n = len(CONCEPTS)
    _, eval_idx = A.stratified_split(CONCEPTS, 0.60, 20260903)
    mask = np.zeros(n, bool)
    mask[eval_idx] = True

    fired = collections.Counter()
    for seed in range(SEEDS):
        rng = np.random.default_rng(4242 + seed)
        d = A.cosine_rdm(rng.normal(size=(n, DIMS)))
        for rel, pairs in sorted(RELATIONS.items()):
            observed = A.loo_consistency([A.relation_signature(d, a, b) for a, b in pairs], mask)
            null = []
            for _ in range(PERMUTATIONS):
                if null_kind == "permuted_pairing":
                    np_pairs = A.permuted_pairing_pairs(pairs, rng)
                else:
                    np_pairs = A.random_target_pairs(
                        pairs, np.arange(n), rng,
                        concept_regions=REGIONS, region_matched=True,
                    )
                null.append(A.loo_consistency([A.relation_signature(d, a, b) for a, b in np_pairs], mask))
            if A.p_greater(observed, null) <= 0.05:
                fired[rel] += 1

    bad = {r: c for r, c in fired.items() if c > MAX_FALSE_POSITIVES}
    assert not bad, (
        f"{null_kind} null is miscalibrated on random geometry: "
        f"{bad} of {SEEDS} runs at p<=.05 (expected <= {MAX_FALSE_POSITIVES})"
    )


def test_no_two_concepts_share_a_surface_form():
    """A shared string makes two concepts identical by construction.

    `learn` and `study` both carried the Chinese term 学习, so in every Chinese
    system their representations were byte-identical: a zero distance that says
    nothing about meaning, sitting inside the RDM that the whole cross-language
    comparison is built on. Found by QC on the first real extraction, not by
    reading the benchmark.
    """
    for lang in ("en", "zh"):
        seen = collections.Counter(r[lang].strip() for r in CONCEPTS)
        dupes = {k: v for k, v in seen.items() if v > 1}
        assert not dupes, f"{lang} surface forms shared by multiple concepts: {dupes}"


def test_every_concept_has_both_languages_and_a_region():
    for r in CONCEPTS:
        assert r["en"].strip(), r
        assert r["zh"].strip(), r
        assert r["region"].strip(), r
