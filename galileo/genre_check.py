#!/usr/bin/env python3
"""Measure surface register across corpora, so 'different genre' is not mistaken for 'different meaning'.

The base-model arm has a confound it cannot design away. A base model has never
been taught to follow instructions, so it cannot be asked to write in a genre; it
can only be given an opener and left to continue. Shown examples it would imitate
them, but the only examples available are human posts (which the no-human-posts
control forbids) or tuned-model posts (which would contaminate the comparison).
So if the base corpus turns out to sit apart from the tuned models, that could
mean instruction tuning produces the convergence -- or merely that the base model
wrote in a different genre.

This does not resolve that, but it does make it visible and quantitative rather
than an unexamined caveat. These are surface features that track register and not
subject matter, so a corpus that is an outlier here is one whose distance from the
others has a plausible stylistic explanation that must be reported alongside the
semantic result.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

FIRST = re.compile(r"\b(i|we|my|our|me|us|i'm|we're|i've)\b", re.I)
YOU = re.compile(r"\b(you|your|you're)\b", re.I)
NUM = re.compile(r"\d")


def stats(posts):
    words = [len(p.split()) for p in posts]
    return {
        "posts": len(posts),
        "mean_words": statistics.fmean(words),
        "first_person_%": 100 * sum(bool(FIRST.search(p)) for p in posts) / len(posts),
        "second_person_%": 100 * sum(bool(YOU.search(p)) for p in posts) / len(posts),
        "question_%": 100 * sum("?" in p for p in posts) / len(posts),
        "has_number_%": 100 * sum(bool(NUM.search(p)) for p in posts) / len(posts),
        "capitalised_start_%": 100 * sum(p[:1].isupper() for p in posts) / len(posts),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard",
                    default=str(pathlib.Path.home() / "projects/data/exorde/data_49.parquet"))
    ap.add_argument("--corpus", action="append", default=[], metavar="NAME=PATH")
    args = ap.parse_args()

    from text_space import load_posts
    rows = {"people": [p["text"] for p in load_posts(args.shard, theme="Politics")[:6000]]}
    for spec in args.corpus:
        name, _, path = spec.partition("=")
        rows[name] = [json.loads(l)["text"]
                      for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
                      if l.strip()]

    keys = list(stats(rows["people"]))
    print(f"{'corpus':<11s}" + "".join(f"{k:>19s}" for k in keys[1:]))
    S = {}
    for name, posts in rows.items():
        S[name] = stats(posts)
        print(f"{name:<11s}" + "".join(f"{S[name][k]:>19.1f}" for k in keys[1:]))

    # Flag any corpus that is a clear outlier on register. Reported, never used to
    # exclude anything -- the point is to say plainly which differences might be
    # style rather than meaning.
    print("\nregister outliers (>2 SD from the mean of the others, per feature):")
    found = False
    for k in keys[1:]:
        for name in S:
            others = [S[o][k] for o in S if o != name]
            m, sd = statistics.fmean(others), (statistics.pstdev(others) or 1e-9)
            if abs(S[name][k] - m) > 2 * sd:
                print(f"  {name:<10s} {k:<20s} {S[name][k]:.1f} vs {m:.1f} in the others")
                found = True
    if not found:
        print("  none -- no corpus stands out on surface register")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
