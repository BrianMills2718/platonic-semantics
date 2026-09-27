#!/usr/bin/env python3
"""Build a Galileo-shaped semantic space from text instead of from survey answers.

Arm 1 asks a respondent for distances directly. This is the instrument arms 2 and
3 share: distances *inferred* from how concepts are used, so the same measurement
applies to human writing and to model writing, and any difference between those
two is attributable to the writer rather than to the question.

The lineage is the one the RAND chapter names beside Galileo -- semantic network
analysis, where "connections between words strengthened weakened depending
whether words co-occur same window". Two concepts here are close when they occur
in *similar company*, not merely when they occur together: a second-order measure
over PPMI context profiles. First-order co-occurrence would call `vaccine` and
`hesitancy` close because they appear side by side, which is association rather
than semantic proximity, and would put every frequent word near every other.

What this deliberately gives up: ratio scale. Woelfel's rod makes two
respondents' numbers comparable without alignment. Text-derived distances have no
rod, so cross-instrument comparison with arm 1 falls back to correlating
structures. Arms 2 and 3 remain directly comparable to each other because they
share this one instrument, which is what their comparison needs.

Two constraints are load-bearing and are enforced rather than assumed:

* **A shared vocabulary.** Both groups are described over the *same* concept set,
  exactly as Galileo's respondents all rate the same attitude objects. Letting
  each group pick its own vocabulary would produce two incomparable spaces and a
  difference that means nothing.
* **Matched corpus size.** PPMI is sensitive to how much text it saw, so a group
  with more posts gets sharper estimates and would look artificially coherent.
  Groups are downsampled to equal post counts.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import re
import statistics

import numpy as np
from scipy.stats import rankdata
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent

# Function words carry syntax, not position in a semantic space. Kept explicit and
# short rather than importing a stopword list, so the exclusions are auditable.
STOP = set("""a an the and or but if while of to in on at by for with from as is are was were be been
being do does did have has had having i you he she it we they me him her them my your his its our their
this that these those there here what which who whom whose when where why how all any both each few more
most other some such no nor not only own same so than too very can will just should now s t don t re ve
ll d m o y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren
won wouldn about into over under again further then once up down out off above below between through
during before after get got go going one two get like know think people said say says also would could
because even time want need against years make see never still way back good many new much take every
right thing things lot bad better best real actually literally maybe probably sure yeah okay gonna
let make made makes making come comes coming came give gives given put puts keep keeps kept
tell tells told ask asks asked look looks looked feel feels felt seem seems seemed
day days week weeks month months year today tomorrow yesterday
it's it's he's she's they're we're you're i'm that's there's here's what's who's isn't wasn't aren't weren't
doesn't didn't don't won't wouldn't couldn't shouldn't can't cannot ain't
yes no not none nothing something anything everything someone anyone everyone nobody
first last next another whole full half less least lately soon far near away
stop start end begin done use used using try tries tried
""".split())

TOKEN = re.compile(r"[a-z][a-z'\-]{2,}")


def tokenize(text: str) -> list[str]:
    return [w for w in TOKEN.findall(text.lower()) if w not in STOP and len(w) > 2]


def load_posts(shard, theme=None, language="en", limit=None):
    f = pq.ParquetFile(shard)
    cols = ["original_text", "author_hash", "language", "primary_theme", "sentiment"]
    out = []
    for batch in f.iter_batches(batch_size=50000, columns=cols):
        d = batch.to_pydict()
        for txt, auth, lang, th, sent in zip(d["original_text"], d["author_hash"],
                                             d["language"], d["primary_theme"], d["sentiment"]):
            if not txt or (language and lang != language):
                continue
            if theme and th != theme:
                continue
            out.append({"text": txt, "author": auth, "theme": th,
                        "sentiment": float(sent) if sent is not None else None})
            if limit and len(out) >= limit:
                return out
    return out


def shared_vocabulary(groups, size):
    """Concepts frequent in EVERY group, so no group is described in another's terms."""
    per_group = []
    for posts in groups:
        c = collections.Counter()
        for p in posts:
            c.update(set(tokenize(p["text"])))
        per_group.append(c)
    common = set(per_group[0])
    for c in per_group[1:]:
        common &= set(c)
    scored = sorted(common, key=lambda w: min(c[w] for c in per_group), reverse=True)
    return scored[:size]


def ppmi_profiles(posts, vocab, window, max_context=2000):
    """PPMI of each concept against a bounded set of context words, over a sliding window.

    `max_context` is load-bearing, not a tuning knob. Left unbounded, a profile has
    one dimension per word that ever appeared near a concept -- tens of thousands,
    the great majority seen once or twice. Those columns carry no reliable signal
    but do enter the cosine, so the measured distance between two concepts is
    dominated by which rare words happened to fall near them in this sample. That
    is exactly the noise the split-half check was failing on. Keeping only the most
    frequent context words is the standard remedy in distributional semantics.
    """
    idx = {w: i for i, w in enumerate(vocab)}
    ctx_counts = collections.Counter()
    for p in posts:
        ctx_counts.update(tokenize(p["text"]))
    keep = {w for w, _ in ctx_counts.most_common(max_context)}
    ctx_index: dict[str, int] = {}
    pairs = collections.Counter()
    concept_tot = collections.Counter()
    ctx_tot = collections.Counter()
    total = 0
    for p in posts:
        toks = tokenize(p["text"])
        for i, w in enumerate(toks):
            if w not in idx:
                continue
            lo, hi = max(0, i - window), min(len(toks), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                c = toks[j]
                if c not in keep:
                    continue
                ci = ctx_index.setdefault(c, len(ctx_index))
                pairs[(idx[w], ci)] += 1
                concept_tot[idx[w]] += 1
                ctx_tot[ci] += 1
                total += 1
    if total == 0:
        raise RuntimeError("no co-occurrences found; vocabulary and corpus do not overlap")
    M = np.zeros((len(vocab), len(ctx_index)), dtype=np.float32)
    for (wi, ci), n in pairs.items():
        pmi = math.log((n * total) / (concept_tot[wi] * ctx_tot[ci]))
        if pmi > 0:                       # positive PMI only: negatives are unreliable at this scale
            M[wi, ci] = pmi
    return M


def distance_matrix(M):
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    Mn = M / np.maximum(norm, 1e-12)
    d = 1.0 - Mn @ Mn.T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, None)


def spearman(a, b):
    # Average ranks for ties. argsort(argsort(x)) hands tied values arbitrary
    # distinct ranks, which biases rho and makes it depend on input order.
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    ra -= ra.mean(); rb -= rb.mean()
    den = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else float("nan")


def spearman_brown(r):
    """Correct a split-half correlation up to the full corpus length.

    Each half sees only half the text, so a split-half figure is the reliability
    of a HALF-SIZED corpus -- but the cross-corpus comparison it is the ceiling
    for is computed on full corpora. Left uncorrected it understates the ceiling
    badly, to the point where two different corpora were seen to agree with each
    other MORE than a corpus agreed with itself, which is not a possible state of
    affairs and is what exposed the error. Spearman-Brown is the standard
    psychometric correction for exactly this.
    """
    return 2 * r / (1 + r) if r > -1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard",
                    default=str(pathlib.Path.home() / "projects/data/exorde/data_49.parquet"),
                    help="durable path -- a session scratchpad is wiped between sessions "
                         "and took this corpus with it once")
    ap.add_argument("--theme", default="Politics")
    ap.add_argument("--vocab-size", type=int, default=60)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--out", default="results/text_space_politics.json")
    args = ap.parse_args()

    posts = load_posts(args.shard, theme=args.theme)
    with_s = [p for p in posts if p["sentiment"] is not None]
    print(f"{args.theme}, English: {len(posts):,} posts   ({len(with_s):,} with sentiment)")

    # Same topic, opposed stance -- the analogue of Woelfel's Bush vs Dukakis
    # respondents. Splitting by theme instead would compare different vocabularies
    # and show only that different people discuss different things.
    vals = sorted(p["sentiment"] for p in with_s)
    lo_cut, hi_cut = vals[len(vals) // 4], vals[3 * len(vals) // 4]
    neg = [p for p in with_s if p["sentiment"] <= lo_cut]
    pos = [p for p in with_s if p["sentiment"] >= hi_cut]

    n = min(len(neg), len(pos))
    rng = random.Random(20260905)
    neg, pos = rng.sample(neg, n), rng.sample(pos, n)
    print(f"negative-sentiment group: {len(neg):,} posts   (sentiment <= {lo_cut:.2f})")
    print(f"positive-sentiment group: {len(pos):,} posts   (sentiment >= {hi_cut:.2f})")
    print("  corpora downsampled to equal size so PPMI sharpness cannot differ by group")

    vocab = shared_vocabulary([neg, pos], args.vocab_size)
    print(f"\nshared vocabulary ({len(vocab)}): {', '.join(vocab[:18])} ...")

    D = {}
    for name, grp in (("negative", neg), ("positive", pos)):
        D[name] = distance_matrix(ppmi_profiles(grp, vocab, args.window))

    iu = np.triu_indices(len(vocab), 1)
    agree = spearman(D["negative"][iu], D["positive"][iu])

    # The control without which the cross-group number is uninterpretable. PPMI
    # cosine distances at this vocabulary size sit in a narrow band near 0.9, so a
    # correlation of 0.675 could equally be two views of one structure seen through
    # sampling noise. Splitting each group in half and measuring agreement WITHIN it
    # gives the ceiling that sampling alone permits: cross-group agreement is only
    # evidence of a real difference if it falls clearly below that.
    halves = {}
    for name, grp in (("negative", neg), ("positive", pos)):
        g = list(grp); rng.shuffle(g)
        mid = len(g) // 2
        a = distance_matrix(ppmi_profiles(g[:mid], vocab, args.window))
        b = distance_matrix(ppmi_profiles(g[mid:], vocab, args.window))
        halves[name] = spearman(a[iu], b[iu])

    ceiling = statistics.fmean(halves.values())
    print(f"\n  within-group (split-half) agreement:")
    for k, v in halves.items():
        print(f"    {k:9s} rho = {v:.3f}   <- ceiling set by sampling noise alone")
    print(f"  cross-group agreement:  rho = {agree:.3f}")
    gap = ceiling - agree
    print(f"  gap below ceiling:      {gap:+.3f}"
          f"   {'-> a real difference between the groups' if gap > 0.05 else '-> NOT distinguishable from sampling noise'}")

    diff = D["negative"] - D["positive"]
    pair_scores = sorted(
        ((abs(diff[i, j]), vocab[i], vocab[j], D['negative'][i, j], D['positive'][i, j])
         for i, j in zip(*iu)), reverse=True)
    print("\nconcept pairs the two groups place most differently:")
    for gap, a, b, dn, dp in pair_scores[:10]:
        closer = "negative" if dn < dp else "positive"
        print(f"  {a:>14s} — {b:<14s}  neg {dn:.3f}  pos {dp:.3f}   ({closer} group holds them closer)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "theme": args.theme, "vocab": vocab, "window": args.window,
        "group_sizes": {"negative": len(neg), "positive": len(pos)},
        "sentiment_cuts": {"negative_max": float(lo_cut), "positive_min": float(hi_cut)},
        "cross_group_rho": float(agree), "split_half_rho": {k: float(v) for k, v in halves.items()},
        "ceiling": float(ceiling), "gap_below_ceiling": float(gap),
        "distances": {k: v.tolist() for k, v in D.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
