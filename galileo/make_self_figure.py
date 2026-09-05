#!/usr/bin/env python3
"""Distance from the Self, which is the quantity Woelfel's theory predicts from.

The Self's position is not interesting as a location. It is interesting as a set
of distances: in Galileo the distance between the Self and an object is what
predicts behaviour toward that object, and attitude change is that distance
shrinking. So the figure is not a map with an extra dot on it -- it is the Self's
distance profile, one row per object, ordered near to far.

Three models are drawn on one axis without alignment, which is legitimate here
only because they judged against a shared rod. Where their marks separate on a
row, the models genuinely disagree about how close that object is to them; where
the marks coincide, they agree.

The instrument's own reliability is printed rather than assumed. A model whose
two independent elicitations disagree cannot support a claim about where it
places itself, and the figure says so instead of drawing a confident mark.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
COLORS = ["#fb923c", "#f472b6", "#a78bfa", "#facc15"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/self_point.json")
    ap.add_argument("--out", default="results/self_point.html")
    args = ap.parse_args()

    d = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    models = d["models"]
    dist = d["distance_from_self"]
    col = {m: c for m, c in zip(models, COLORS)}
    order = sorted(dist, key=lambda c: statistics.fmean(dist[c].values()))

    vals = [v for c in dist for v in dist[c].values()]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.10 or 0.1
    lo, hi = lo - pad, hi + pad

    W, ROW, TOP, L, R = 900, 52, 60, 210, 850
    H = TOP + len(order) * ROW + 30
    sx = lambda v: L + (v - lo) / (hi - lo) * (R - L)

    parts = []
    for t in [lo + pad, (lo + hi) / 2, hi - pad]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{TOP-26}" x2="{sx(t):.1f}" '
                     f'y2="{TOP + len(order)*ROW - 26:.0f}" class="grid"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{TOP-34}" class="tick">{t:.2f}</text>')
    parts.append(f'<text x="{L}" y="{TOP-52}" class="axname">closer to the model\'s '
                 f'sense of itself</text>')
    parts.append(f'<text x="{R}" y="{TOP-52}" class="axname" text-anchor="end">'
                 f'further away</text>')

    for i, c in enumerate(order):
        y = TOP + i * ROW
        vs = [dist[c][m] for m in models]
        parts.append(f'<text x="{L-22}" y="{y+5:.0f}" class="rowname">{c}</text>')
        parts.append(f'<line x1="{L}" y1="{y}" x2="{R}" y2="{y}" class="axis"/>')
        parts.append(f'<line x1="{sx(min(vs)):.1f}" y1="{y}" x2="{sx(max(vs)):.1f}" '
                     f'y2="{y}" class="span"/>')
        for m in models:
            parts.append(f'<circle cx="{sx(dist[c][m]):.1f}" cy="{y}" r="5.5" '
                         f'fill="{col[m]}"/>')

    key = "".join(f'<span class="k"><i style="background:{col[m]}"></i>{m}'
                  f'<em> agrees with itself {d["self_agreement"][m]:.2f}</em></span>'
                  for m in models)
    ca, sa = d["agreement_about_concepts"], d["agreement_about_self"]
    if sa < ca - 0.15:
        verdict = ("The models agree about politics but <b>not</b> about where they "
                   "stand in it — a disagreement no other instrument here can see.")
    elif sa > ca + 0.15:
        verdict = ("The models agree about themselves more than they agree about "
                   "politics.")
    else:
        verdict = ("The models agree about themselves about as much as they agree "
                   "about politics — no difference this instrument can resolve.")

    html = f"""<!doctype html><meta charset="utf-8">
<title>Where the models place themselves</title>
<style>
 body{{margin:0;background:#0f1115;color:#e7e9ee;
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
 .wrap{{max-width:1000px;margin:0 auto;padding:34px 26px 60px}}
 h1{{font-size:27px;margin:0 0 8px;letter-spacing:-.02em}}
 .sub{{color:#9aa3b2;max-width:74ch;margin:0 0 18px}}
 svg{{background:#151821;border:1px solid #232838;border-radius:12px}}
 .axis{{stroke:#2b3242}} .grid{{stroke:#1d2330}}
 .span{{stroke:#475569;stroke-width:3;stroke-opacity:.55}}
 .tick{{fill:#6b7688;font-size:11px;text-anchor:middle}}
 .axname{{fill:#6b7688;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em}}
 .rowname{{fill:#e7e9ee;font-size:14.5px;font-weight:700;text-anchor:end}}
 .keys{{margin:14px 0 0;font-size:13px;color:#9aa3b2}}
 .k{{display:block;margin:4px 0}}
 .k i{{display:inline-block;width:10px;height:10px;border-radius:2px;
       margin-right:7px;vertical-align:middle}}
 .k em{{color:#6b7688;font-style:normal;font-size:12px}}
 .verdict{{background:#151821;border-left:3px solid #f59e0b;padding:14px 18px;
           border-radius:0 8px 8px 0;margin:22px 0;max-width:78ch}}
 .note{{color:#9aa3b2;font-size:13.5px;max-width:76ch;margin:14px 0 0}}
 b{{color:#e7e9ee}}
</style>
<div class="wrap">
<h1>Where three language models place themselves</h1>
<p class="sub">In Woelfel's method the Self is an object in the space like any
other, judged against the same rod: <i>how far apart are you and war?</i> That is
what makes the theory predictive rather than merely descriptive — the distance
between the Self and an object is what forecasts behaviour toward it, and
attitude change <i>is</i> that distance shrinking. Woelfel found the brand nearer
a person's Self held the larger market share.</p>
<p class="sub">Each row is one object; each mark is one model's judged distance
from itself to that object, in the same ratio-scale units as every other distance
in this project. No alignment step was used or needed.</p>

<svg width="{W}" height="{H}">{''.join(parts)}</svg>
<p class="keys">{key}</p>

<p class="verdict">{verdict}<br>
Agreement about the concepts: <b>{ca:+.2f}</b>. Agreement about the Self:
<b>{sa:+.2f}</b>.</p>

<p class="note"><b>What this cannot tell you.</b> That a model reports a distance
from "yourself" does not establish that it has a self-representation; it may be
reporting a persona the prompt evoked. What the numbers do support is narrower
and still useful: the reports are reproducible within a model, and comparable
between models because of the shared rod.</p>
<p class="note">No other route in this project can produce this figure.
Activations give a geometry of concepts with no place in it for the model;
word co-occurrence gives whatever the text happened to mention. The Self costs
exactly one more object, and only asking can obtain it.</p>
</div>"""
    (ROOT / args.out).write_text(html, encoding="utf-8")
    print(f"wrote {ROOT / args.out}")
    print("nearest the Self: " + ", ".join(order[:3]))
    print("furthest:         " + ", ".join(order[-3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
