#!/usr/bin/env python3
"""The recovered dimension: semantic geometry lines political groups up by sentiment.

Five groups of people, split only by the sentiment of their political writing,
were each measured over the same ten concepts. Nothing in the instrument knows
about sentiment -- it sees only which words occur near which. Yet the space it
recovers orders the groups along the sentiment scale.

A 2-D map is the wrong figure for this and was tried first: the three middle
bands pile up on one another and their labels collide, which shows that they are
close without showing the thing that matters. The finding is an *ordering*, so
this draws the ordering: each group's position on the first recovered axis, with
the error bar that says how far it moves when its own corpus is split in half.

The claim is tested rather than eyeballed. The five bands admit 120 possible
orderings; the true sentiment order achieves the highest rank correlation with
semantic distance of any of them, and only it and its mirror reach that value
(p = 0.017, exact). That is reported on the figure, because a monotone-looking
staircase through five points is exactly the shape noise produces often enough
to be worth guarding against.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np

from text_space import spearman

ROOT = pathlib.Path(__file__).resolve().parent


def permutation_p(D):
    """Exact test over every ordering of the bands. Small n, so enumerate."""
    n = D.shape[0]
    pairs = list(itertools.combinations(range(n), 2))

    def corr(order):
        pos = {g: k for k, g in enumerate(order)}
        band = np.array([abs(pos[i] - pos[j]) for i, j in pairs], float)
        sem = np.array([D[i, j] for i, j in pairs], float)
        return spearman(band, sem)

    obs = corr(list(range(n)))
    null = [corr(list(p)) for p in itertools.permutations(range(n))]
    return obs, sum(1 for v in null if v >= obs) / len(null), max(null)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/human_only_space.json")
    ap.add_argument("--out", default="results/sentiment_dimension.html")
    args = ap.parse_args()

    d = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    names = d["respondents"]
    P = np.array(d["coords"])
    err = d["error_radius"]
    ceil = d["ceiling"]
    D = np.array(d["disagreement"])
    obs, p, best = permutation_p(D)

    x = P[:, 0]
    n = len(names)
    lo = min(x[i] - err[names[i]] for i in range(n))
    hi = max(x[i] + err[names[i]] for i in range(n))
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    W, ROW, TOP, L, R = 900, 74, 52, 190, 860
    H = TOP + n * ROW + 40
    sx = lambda v: L + (v - lo) / (hi - lo) * (R - L)

    parts = [f'<line x1="{L}" y1="{TOP-18}" x2="{R}" y2="{TOP-18}" class="axis"/>']
    for t in np.linspace(lo + pad, hi - pad, 5):
        parts.append(f'<line x1="{sx(t):.1f}" y1="{TOP-24}" x2="{sx(t):.1f}" '
                     f'y2="{TOP + n*ROW - 30}" class="grid"/>')
    for i, nm in enumerate(names):
        y = TOP + i * ROW
        a, b = sx(x[i] - err[nm]), sx(x[i] + err[nm])
        parts.append(f'<text x="{L-22}" y="{y+5:.0f}" class="rowname">{nm}</text>')
        parts.append(f'<text x="{L-22}" y="{y+19:.0f}" class="sub">self {ceil[nm]:.2f}</text>')
        parts.append(f'<line x1="{a:.1f}" y1="{y}" x2="{b:.1f}" y2="{y}" class="bar"/>')
        parts.append(f'<line x1="{a:.1f}" y1="{y-6}" x2="{a:.1f}" y2="{y+6}" class="cap"/>')
        parts.append(f'<line x1="{b:.1f}" y1="{y-6}" x2="{b:.1f}" y2="{y+6}" class="cap"/>')
        parts.append(f'<circle cx="{sx(x[i]):.1f}" cy="{y}" r="6" class="pt"/>')
    # The trend line the ordering claim is about.
    parts.append('<polyline points="' + " ".join(
        f"{sx(x[i]):.1f},{TOP + i*ROW}" for i in range(n)) + '" class="trend"/>')

    html = f"""<!doctype html><meta charset="utf-8">
<title>The recovered dimension</title>
<style>
 body{{margin:0;background:#0f1115;color:#e7e9ee;
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
 .wrap{{max-width:1000px;margin:0 auto;padding:34px 26px 60px}}
 h1{{font-size:26px;margin:0 0 8px;letter-spacing:-.02em}}
 .sub-p{{color:#9aa3b2;max-width:74ch;margin:0 0 20px}}
 svg{{background:#151821;border:1px solid #232838;border-radius:12px}}
 .axis{{stroke:#2b3242}} .grid{{stroke:#1d2330}}
 .bar{{stroke:#3b82f6;stroke-width:2.5;stroke-opacity:.55}}
 .cap{{stroke:#3b82f6;stroke-width:2;stroke-opacity:.55}}
 .pt{{fill:#60a5fa}}
 .trend{{fill:none;stroke:#f59e0b;stroke-width:2;stroke-dasharray:5 4;stroke-opacity:.8}}
 .rowname{{fill:#e7e9ee;font-size:14px;font-weight:700;text-anchor:end}}
 .sub{{fill:#6b7688;font-size:11px;text-anchor:end}}
 .stat{{background:#151821;border-left:3px solid #f59e0b;padding:13px 18px;
        border-radius:0 8px 8px 0;margin:20px 0;max-width:78ch}}
 .note{{color:#9aa3b2;font-size:13.5px;max-width:76ch;margin:16px 0 0}}
 b{{color:#e7e9ee}}
</style>
<div class="wrap">
<h1>Semantic geometry recovers the political sentiment scale</h1>
<p class="sub-p">Five groups of people, separated only by how positive or
negative their political posts are. Each was measured over the same ten
concepts, on {d['matched_content_tokens']:,} words each. <b>The instrument
never sees sentiment</b> — it sees only which words occur near which. Yet the
space it recovers lines the groups up in sentiment order, top to bottom.</p>

<svg width="{W}" height="{H}">{''.join(parts)}</svg>

<p class="stat">Of the <b>120</b> possible orderings of these five groups, the
true sentiment order gives the strongest relationship between position and
semantic distance — <b>{obs:.2f}</b>, the highest any ordering achieves. Only it
and its mirror reach that value: <b>p = {p:.3f}</b>, exact.</p>

<p class="note">Bars are not confidence intervals in the usual sense: each is how
far that group's point moves when its own corpus is split in half and placed
twice, so it is displacement from sampling alone. <b>The two middle groups
overlap heavily and swap order</b> — they are 0.016 apart with bars of 0.26 and
0.39, so nothing should be read into which of them comes first. The ordering
claim rests on the whole sequence, which is what the permutation test above
measures, not on any adjacent pair.</p>
<p class="note">Reliability is high here (each group agrees with itself at
0.83–0.89) because these groups were measured on the full human corpus rather
than being cut down to match a smaller model corpus. In the combined
people-and-models map the same groups are starved to 51,698 words and their
self-agreement falls to 0.47–0.71, which is why their differences are not
resolvable there.</p>
</div>"""
    (ROOT / args.out).write_text(html, encoding="utf-8")
    print(f"wrote {ROOT / args.out}")
    print(f"observed rho {obs:.3f}  best-possible {best:.3f}  p={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
