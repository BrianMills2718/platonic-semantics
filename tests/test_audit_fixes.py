"""Regression tests for three correctness defects found in an external audit.

1. `elicit_averaged` must survive a permutation that omits a few judgments and
   recover the missing pair from the other permutations -- the policy stated in
   `schema_for` -- instead of dying on the first short answer.
2. A templated prompt mode (e.g. `neutral`) without `--average-modes` must pool
   only the term's own tokens, not the template's words.
3. Spearman correlation must give tied values their average rank. Ranking ties
   by `argsort(argsort(x))` gives them arbitrary distinct ranks, which inflated
   the reported sentiment-ordering rho from 0.680 to 0.794.

None of these touch a network, a model download, or a GPU: the LLM call, torch
and transformers are replaced by fakes.
"""
from __future__ import annotations

import importlib
import itertools
import random
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def _fresh_import(monkeypatch, directory, name, stubs=()):
    """Import `name` from `directory` with its external deps replaced by fakes.

    Always stubbed, not only when absent: an installed `llm_client` of a
    different version need not export the names these scripts import, and the
    code under test never reaches the real call anyway.
    """
    monkeypatch.syspath_prepend(str(directory))
    for stub in stubs:
        monkeypatch.setitem(sys.modules, stub, mock.MagicMock(name=stub))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


# --- 1. elicit_averaged ------------------------------------------------------

def _elicit_module(monkeypatch, truth, drop):
    """`elicit` with a fake model that answers truthfully but omits some pairs.

    `drop(pairs)` returns the 1-based list positions the fake model leaves out.
    """
    el = _fresh_import(monkeypatch, ROOT / "galileo", "elicit",
                       stubs=("llm_client", "llm_client.prompts"))
    monkeypatch.setattr(el, "render_prompt", lambda _p, **kw: {"pairs": kw["pairs"]})

    def fake_call(model, messages, schema, **kw):
        pairs = messages["pairs"]
        skip = set(drop(pairs))
        return ({"distances": [{"pair": i, "distance": truth[p]}
                               for i, p in enumerate(pairs, 1) if i not in skip]},
                None)

    monkeypatch.setattr(el, "call_llm_json_schema", fake_call)
    return el


CONCEPTS = ["a", "b", "c", "d", "e", "f", "g", "h"]
PAIRS = list(itertools.combinations(CONCEPTS, 2))          # 28 pairs
TRUTH = {p: float(10 + 3 * k) for k, p in enumerate(PAIRS)}


def test_short_answer_is_recovered_across_permutations(monkeypatch):
    # 27 of 28 every call: schema-valid, but one judgment is always missing.
    el = _elicit_module(monkeypatch, TRUTH, drop=lambda pairs: [len(pairs)])
    mean, per_perm, raw_mean = el.elicit_averaged(
        "fake", PAIRS, ("x", "y", 100), "t", 1.0, 8, random.Random(0))
    assert len(mean) == len(PAIRS) and None not in mean and None not in raw_mean
    # Raw values are exact averages of the truth wherever observed.
    assert raw_mean == pytest.approx([TRUTH[p] for p in PAIRS])
    # Each kept permutation reports its gap as None, not as a fabricated 0.
    assert all(sum(v is None for v in perm) == 1 for perm in per_perm)


def test_pair_missing_from_every_permutation_fails_loudly(monkeypatch):
    lost = PAIRS[5]
    el = _elicit_module(monkeypatch, TRUTH,
                        drop=lambda pairs: [pairs.index(lost) + 1])
    with pytest.raises(RuntimeError, match="missing from every"):
        el.elicit_averaged("fake", PAIRS, ("x", "y", 100), "t", 1.0, 4,
                           random.Random(0))


def test_thin_permutation_is_dropped(monkeypatch):
    calls = {"n": 0}

    def drop(pairs):
        calls["n"] += 1
        # First call returns only 3 of 28 judgments: too thin to rescale.
        return list(range(4, len(pairs) + 1)) if calls["n"] == 1 else []

    el = _elicit_module(monkeypatch, TRUTH, drop=drop)
    _, per_perm, _ = el.elicit_averaged("fake", PAIRS, ("x", "y", 100), "t",
                                        1.0, 3, random.Random(0))
    assert len(per_perm) == 2


# --- 2. term-span pooling for templated prompts -------------------------------

def _extract_module(monkeypatch):
    for heavy in ("torch", "transformers"):
        fake = mock.MagicMock(name=heavy)
        if heavy == "transformers":
            fake.__version__ = "4.45.0"
        monkeypatch.setitem(sys.modules, heavy, fake)
    return _fresh_import(monkeypatch, ROOT / "src", "extract_representations")


def _run_extract(monkeypatch, tmp_path, prompt_mode):
    er = _extract_module(monkeypatch)
    calls = {"all": [], "term": []}

    def fake_all(tok, model, texts, *a):
        calls["all"].append(texts)
        return np.ones((len(texts), 2, 3))

    def fake_term(tok, model, template, terms, *a):
        calls["term"].append((template, terms))
        return np.ones((len(terms), 2, 3))

    monkeypatch.setattr(er, "encode_all", fake_all)
    monkeypatch.setattr(er, "encode_term_in_context", fake_term)
    monkeypatch.setattr(er, "load_model", lambda *a: (None, None, "float32"))
    import json
    key = next(iter(json.loads((ROOT / "experiment_config.json").read_text())["models"]))
    monkeypatch.setattr(sys, "argv", [
        "extract", "--config", str(ROOT / "experiment_config.json"),
        "--concepts", str(ROOT / "benchmark" / "concepts.csv"),
        "--outdir", str(tmp_path), "--models", key, "--languages", "en",
        "--prompt-mode", prompt_mode, "--device", "cpu"])
    er.main()
    return calls


def test_neutral_mode_pools_only_the_term(monkeypatch, tmp_path):
    calls = _run_extract(monkeypatch, tmp_path, "neutral")
    assert calls["all"] == [], "template words were pooled into the representation"
    assert [t for t, _ in calls["term"]] == ["The concept is {term}."]


def test_bare_mode_is_unchanged(monkeypatch, tmp_path):
    calls = _run_extract(monkeypatch, tmp_path, "bare")
    assert calls["term"] == [] and len(calls["all"]) == 1


# --- 3. Spearman with ties ----------------------------------------------------

def _spearmans(monkeypatch):
    g = ROOT / "galileo"
    stubs = ("pyarrow", "pyarrow.parquet", "llm_client", "llm_client.prompts")
    return {name: _fresh_import(monkeypatch, g, name, stubs).spearman
            for name in ("text_space", "elicit_map", "lab_origin")}


def test_spearman_averages_tied_ranks(monkeypatch):
    # Band distances between five ordered groups: heavily tied by construction,
    # exactly the input `make_sentiment_figure` feeds this function.
    idx = list(itertools.combinations(range(5), 2))
    band = np.array([abs(i - j) for i, j in idx], float)
    sem = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.6, 0.7, 0.4, 0.8, 0.35])
    want = spearmanr(band, sem).statistic
    for name, fn in _spearmans(monkeypatch).items():
        assert fn(band, sem) == pytest.approx(want), name
        # And the answer must not depend on the order the pairs are listed in.
        perm = np.random.default_rng(1).permutation(len(band))
        assert fn(band[perm], sem[perm]) == pytest.approx(want), name
