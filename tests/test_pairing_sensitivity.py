"""A relation test must distinguish a relation from an arbitrary pair of concepts.

Two statistics failed that here on 2026-09-04, and each failure is locked in
below so neither is promoted back to primary.

1. The mean-signature convergence in `relation_convergence.csv` is algebraically
   invariant to which source is paired with which target, so it cannot test a
   transformation at all.
2. Its first replacement -- matched versus mismatched pair signatures under a
   *pairing-permutation* null -- was pairing-sensitive but not relation-specific.
   Random non-relational concept pairs cleared it at p = 0.001 with every system
   pair positive, scoring 0.40-0.48 against real relations' 0.18-0.33. It was
   re-measuring first-order RDM agreement one pair at a time.

The separation-matched null is what separates a relation from two concepts that
happen to sit that far apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_semantic_geometry import (  # noqa: E402
    correspondence_effect,
    cosine_rdm,
    masked_cos,
    pair_signature_matrix,
    separation_matched_random_pairs,
)

N_CONCEPTS = 80
N_PAIRS = 14
DIM = 32


def _shared_geometry(rng, n_systems=3, noise=0.35):
    """Systems that genuinely agree about concepts, which is the run-001 situation."""
    base = rng.normal(size=(N_CONCEPTS, DIM)).astype(np.float32)
    return {
        f"sys{i}": cosine_rdm(base + noise * rng.normal(size=base.shape).astype(np.float32))
        for i in range(n_systems)
    }


def _disjoint_pairs(rng):
    picked = rng.choice(N_CONCEPTS, size=2 * N_PAIRS, replace=False)
    return list(zip(picked[:N_PAIRS].tolist(), picked[N_PAIRS:].tolist()))


def _mean_signature_score(rdms, pairs, mask):
    """The retired statistic, with the self-exclusion mask removed."""
    means = {s: np.mean([d[b] - d[a] for a, b in pairs], axis=0) for s, d in rdms.items()}
    keys = list(means)
    return float(np.nanmean([
        masked_cos(means[keys[i]], means[keys[j]], mask)
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
    ]))


def test_mean_signature_statistic_is_pairing_invariant():
    """Why relation_convergence.csv is descriptive only (D17)."""
    rng = np.random.default_rng(20260904)
    rdms = _shared_geometry(rng)
    mask = np.zeros(N_CONCEPTS, dtype=bool)
    mask[rng.choice(N_CONCEPTS, size=N_CONCEPTS // 2, replace=False)] = True

    pairs = _disjoint_pairs(rng)
    observed = _mean_signature_score(rdms, pairs, mask)
    srcs = [a for a, _ in pairs]
    tgts = [b for _, b in pairs]
    for _ in range(10):
        perm = rng.permutation(len(tgts))
        shuffled = [(srcs[i], tgts[perm[i]]) for i in range(len(srcs))]
        assert _mean_signature_score(rdms, shuffled, mask) == pytest.approx(observed, abs=1e-6)


def test_pairing_permutation_alone_is_not_relation_specific():
    """The trap: agreeing systems pass the pairing null on arbitrary pairs.

    This is the check that caught the first replacement statistic. It asserts the
    weakness rather than hiding it, so the pairing null is never reported alone.
    """
    rng = np.random.default_rng(3)
    rdms = _shared_geometry(rng)
    anchors = np.arange(N_CONCEPTS)
    pairs = _disjoint_pairs(rng)

    mats = {s: pair_signature_matrix(d, pairs, anchors) for s, d in rdms.items()}
    observed = correspondence_effect(mats)[0]["matched"]

    null = []
    for _ in range(200):
        perm = rng.permutation(len(pairs))
        shuffled = {s: m[perm] for s, m in mats.items()}
        first = next(iter(shuffled))
        shuffled[first] = mats[first]
        null.append(correspondence_effect(shuffled)[0]["matched"])
    p = (1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1)

    assert p <= 0.01, (
        "arbitrary pairs no longer clear the pairing null; if this starts failing "
        "the pairing null may be usable alone, but confirm on real geometry first"
    )


def test_separation_matched_null_rejects_arbitrary_pairs():
    """The fix: with separation held fixed, arbitrary pairs are not special."""
    rng = np.random.default_rng(5)
    rdms = _shared_geometry(rng)
    anchors = np.arange(N_CONCEPTS)
    mean_dist = np.mean(np.stack(list(rdms.values())), axis=0)
    pairs = _disjoint_pairs(rng)

    observed, _ = correspondence_effect(
        {s: pair_signature_matrix(d, pairs, anchors) for s, d in rdms.items()}
    )
    observed = observed["matched"]

    null = []
    for _ in range(200):
        # ONE draw shared by every system. Drawing per system would give each a
        # different pair set and collapse the null to zero, which reads as an
        # overwhelming result rather than as the mistake it is.
        drawn = separation_matched_random_pairs(pairs, mean_dist, rng)
        null.append(correspondence_effect(
            {s: pair_signature_matrix(d, drawn, anchors) for s, d in rdms.items()}
        )[0]["matched"])
    p = (1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1)

    assert p > 0.05, f"arbitrary pairs called relational at p={p:.3f}"


def test_separation_matched_pairs_match_separation():
    rng = np.random.default_rng(13)
    rdms = _shared_geometry(rng)
    mean_dist = np.mean(np.stack(list(rdms.values())), axis=0)
    pairs = _disjoint_pairs(rng)

    drawn = separation_matched_random_pairs(pairs, mean_dist, rng)
    assert len(drawn) == len(pairs)
    used = [c for p in drawn for c in p]
    assert len(set(used)) == len(used), "a concept was reused across null pairs"
    for (a, b), (x, y) in zip(pairs, drawn):
        assert abs(mean_dist[a, b] - mean_dist[x, y]) <= 0.05 + 1e-6
