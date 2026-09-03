"""The freeze record must describe the artifacts that are actually here.

`docs/EXPERIMENT_0_FREEZE.md` pins SHA-256 hashes of the benchmark and the two
core sources so a later reader can tell whether the run they are looking at used
the frozen design. A freeze record that silently drifts from the tree is worse
than none: it looks like provenance and is not.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
FREEZE = ROOT / "docs" / "EXPERIMENT_0_FREEZE.md"
PROVENANCE = ROOT / "docs" / "DATA_PROVENANCE.md"

FROZEN = [
    "benchmark/concepts.csv",
    "benchmark/relations.csv",
    "src/extract_representations.py",
    "src/analyze_semantic_geometry.py",
]


def _sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def test_freeze_record_matches_the_tree():
    text = FREEZE.read_text(encoding="utf-8")
    for rel in FROZEN:
        digest = _sha(rel)
        assert digest in text, (
            f"{rel} has hash {digest}, which is not in EXPERIMENT_0_FREEZE.md. "
            "Re-freeze deliberately and record why it changed, per that file's "
            "own change rule."
        )


def test_provenance_hashes_match_the_tree():
    text = PROVENANCE.read_text(encoding="utf-8")
    for rel in ("benchmark/concepts.csv", "benchmark/relations.csv"):
        assert _sha(rel) in text, f"{rel} hash is stale in DATA_PROVENANCE.md"


def test_no_superseded_v1_hashes_remain():
    """The v2-handoff hashes describe the uncorrected benchmark and null."""
    stale = {
        "9580f94c617a6341edf6eb7f9611a306bc7452bf3888f0795dc46fd623343f07",
        "7640c2246b7b32c7eda2f9318a3c1c3815d560378bd248949c06414ba8702792",
        "b2299f022c86753418ca0bbcd8ca282d509d76a9872a33313ca5551217205590",
        "d8706f1fed28266443ae0a87e99600010824c62b9f5d694bc312eca88b9a5fd7",
    }
    for doc in (FREEZE, PROVENANCE):
        text = doc.read_text(encoding="utf-8")
        found = stale & set(re.findall(r"[0-9a-f]{64}", text))
        assert not found, f"{doc.name} still pins superseded hashes: {found}"


def test_documented_counts_match_the_benchmark():
    import csv

    concepts = list(csv.DictReader((ROOT / "benchmark/concepts.csv").open(encoding="utf-8")))
    relations = list(csv.DictReader((ROOT / "benchmark/relations.csv").open(encoding="utf-8")))
    provenance = PROVENANCE.read_text(encoding="utf-8")
    assert f"{len(concepts)} concepts" in provenance
    assert f"{len(relations)} directed probe pairs" in provenance
