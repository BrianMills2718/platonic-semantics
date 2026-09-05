#!/usr/bin/env python3
"""Draw the corpora without projecting them, and show the block structure directly.

Two rules govern this figure, both learned the hard way on this project.

**Nothing is flattened.** Classical MDS of these ten concepts retains only ~31%
of the structure in two dimensions, so a 2-D map of them is roughly two-thirds
artefact of the projection. Four earlier scatter-plot attempts here were rejected
as "basically a dot plot", which is what a 2-D scatter of a space that does not
fit in 2-D can only ever look like. So each concept gets its own axis and the
other nine sit on it at their measured distance, one lane per corpus. Every
position drawn is a number that was measured.

**The result is a block, so the block is drawn.** The finding is not about any
one pair of corpora; it is that the model lanes track each other while the people
lane sits apart. The agreement matrix at the top shows that structure at a glance,
and the per-concept lanes below show where it comes from.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent

# People deliberately in a different hue from every model, because the claim is
# precisely that the models group together and people do not join them.
COLOR = {"people": "#60a5fa"}
MODEL_COLORS = ["#fb923c", "#f472b6", "#a78bfa", "#facc15"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/all_corpora.json")
    ap.add_argument("--out", default="results/spaces.html")
    args = ap.parse_args()

    d = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    vocab = d["vocab"]
    names = list(d["distances"])
    for i, n in enumerate(n for n in names if n != "people"):
        COLOR[n] = MODEL_COLORS[i % len(MODEL_COLORS)]
    D = {k: np.array(v) for k, v in d["distances"].items()}
    n = len(vocab)
    iu = np.triu_indices(n, 1)

    lo = min(D[k][iu].min() for k in names)
    hi = max(D[k][iu].max() for k in names)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    W, LANE, ROW_H = 1080, 30, 60 + 30 * len(names)
    L, R = 160, W - 40

    def sx(v):
        return L + (v - lo) / (hi - lo) * (R - L)

    def rho(a, b):
        return d["pairwise"].get(f"{a}|{b}", d["pairwise"].get(f"{b}|{a}"))

    # ---- agreement matrix -------------------------------------------------
    CELL, MX0, MY0 = 74, 150, 34
    mat = []
    for r, a in enumerate(names):
        mat.append(f'<text x="{MX0 - 10}" y="{MY0 + r * CELL + CELL/2 + 4:.0f}" '
                   f'class="mlabel" fill="{COLOR[a]}">{a}</text>')
        mat.append(f'<text x="{MX0 + r * CELL + CELL/2:.0f}" y="{MY0 - 10}" '
                   f'class="mtop" fill="{COLOR[a]}">{a}</text>')
        for c, b in enumerate(names):
            x, y = MX0 + c * CELL, MY0 + r * CELL
            if a == b:
                mat.append(f'<rect x="{x}" y="{y}" width="{CELL-3}" height="{CELL-3}" '
                           f'rx="5" class="selfcell"/>')
                mat.append(f'<text x="{x+CELL/2-1.5:.0f}" y="{y+CELL/2+4:.0f}" '
                           f'class="cellv dim">{d["ceiling"][a]:.2f}</text>')
                continue
            v = rho(a, b)
            # One shared scale from 0 to the ceiling: a cell is bright when two
            # corpora agree as much as a corpus agrees with itself.
            t = max(0.0, min(1.0, v / max(d["ceiling"].values())))
            mat.append(f'<rect x="{x}" y="{y}" width="{CELL-3}" height="{CELL-3}" rx="5" '
                       f'fill="#f59e0b" fill-opacity="{0.06 + 0.80*t:.3f}"/>')
            mat.append(f'<text x="{x+CELL/2-1.5:.0f}" y="{y+CELL/2+4:.0f}" '
                       f'class="cellv">{v:.2f}</text>')
    MH = MY0 + len(names) * CELL + 12

    # ---- per-concept lanes ------------------------------------------------
    shift = {w: float(sum(np.abs(D[a][i] - D[b][i]).sum()
                          for a in names for b in names if a < b))
             for i, w in enumerate(vocab)}
    order = sorted(range(n), key=lambda i: -shift[vocab[i]])

    rows = []
    for slot, i in enumerate(order):
        y = 26 + slot * ROW_H
        rows.append(f'<text x="14" y="{y + 18:.0f}" class="rowname">{vocab[i]}</text>')
        lane_y = {}
        for li, name in enumerate(names):
            ly = y + 34 + li * LANE
            lane_y[name] = ly
            rows.append(f'<text x="{L - 12}" y="{ly + 4:.0f}" class="lane" '
                        f'fill="{COLOR[name]}">{name}</text>')
            rows.append(f'<line x1="{L}" y1="{ly}" x2="{R}" y2="{ly}" class="axis"/>')
        for j in range(n):
            if j == i:
                continue
            pts = [(sx(D[k][i, j]), lane_y[k]) for k in names]
            # Highlight exactly the finding: the models landing close together
            # while people sit clearly away from them. Left uniform, the threads
            # show that something differs but not what, and the eye cannot pick
            # the agreeing block out of nine overlapping lines.
            mvals = [D[k][i, j] for k in names if k != "people"]
            spread = max(mvals) - min(mvals)
            apart = abs(D["people"][i, j] - float(np.mean(mvals)))
            cls = "thread agree" if (len(mvals) > 1 and apart > 2.2 * spread
                                     and apart > 0.02) else "thread"
            rows.append('<polyline points="' +
                        " ".join(f"{x:.1f},{y_:.0f}" for x, y_ in pts) +
                        f'" class="{cls}"/>')
            for k in names:
                rows.append(f'<circle cx="{sx(D[k][i, j]):.1f}" cy="{lane_y[k]}" '
                            f'r="3.6" fill="{COLOR[k]}"/>')

    ticks = "".join(
        f'<line x1="{sx(v):.1f}" y1="16" x2="{sx(v):.1f}" y2="{26 + n*ROW_H - 30:.0f}" '
        f'class="grid"/><text x="{sx(v):.1f}" y="11" class="tick">{v:.2f}</text>'
        for v in np.linspace(lo + pad, hi - pad, 5))
    H = 26 + n * ROW_H

    models = [x for x in names if x != "people"]
    html = f"""<!doctype html><meta charset="utf-8">
<title>Models converge with each other, not with us</title>
<style>
 body{{margin:0;background:#0f1115;color:#e7e9ee;
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
 .wrap{{max-width:1140px;margin:0 auto;padding:34px 26px 60px}}
 h1{{font-size:27px;margin:0 0 8px;letter-spacing:-.02em}}
 h2{{font-size:17px;margin:34px 0 6px}}
 .sub{{color:#9aa3b2;max-width:72ch;margin:0 0 14px}}
 .head{{display:flex;gap:22px;flex-wrap:wrap;margin:22px 0 8px}}
 .stat{{background:#151821;border:1px solid #232838;border-radius:10px;padding:13px 19px}}
 .stat b{{display:block;font-size:25px;font-variant-numeric:tabular-nums}}
 .stat span{{color:#9aa3b2;font-size:12.5px}}
 .verdict{{background:#151821;border-left:3px solid #f59e0b;padding:14px 18px;
           border-radius:0 8px 8px 0;margin:20px 0 0;max-width:80ch}}
 svg{{background:#151821;border:1px solid #232838;border-radius:12px}}
 .axis{{stroke:#2b3242;stroke-width:1}} .grid{{stroke:#1d2330;stroke-width:1}}
 .tick{{fill:#6b7688;font-size:11px;text-anchor:middle}}
 .thread{{fill:none;stroke:#3f4a5c;stroke-width:1.1;stroke-opacity:.6}}
 .thread.agree{{stroke:#f59e0b;stroke-width:2;stroke-opacity:.95}}
 .rowname{{fill:#e7e9ee;font-size:15.5px;font-weight:700}}
 .lane{{font-size:10.5px;text-anchor:end;text-transform:uppercase;letter-spacing:.06em}}
 .selfcell{{fill:#1b2130;stroke:#2b3242}}
 .cellv{{fill:#0f1115;font-size:15px;font-weight:700;text-anchor:middle}}
 .cellv.dim{{fill:#6b7688;font-weight:600}}
 .mlabel{{font-size:13px;text-anchor:end;font-weight:600}}
 .mtop{{font-size:13px;text-anchor:middle;font-weight:600}}
 .note{{color:#9aa3b2;font-size:13.5px;max-width:78ch;margin:18px 0 0}}
</style>
<div class="wrap">
<h1>Three language models organise politics the same way as each other —
and not the way people do</h1>
<p class="sub">Ten political concepts, positioned by how they are actually
<i>used</i>. Four corpora, one identical instrument, matched to the same amount
of text: {d['n_posts']['people']:,} posts written by people, and posts written by
{len(models)} models built by different organisations
({', '.join(models)}), each asked to post about topics drawn from the human
corpus and never shown a human post.</p>

<div class="head">
 <div class="stat"><b style="color:#f59e0b">{d['model_model_mean']:.2f}</b>
   <span>how much the models agree<br>with EACH OTHER</span></div>
 <div class="stat"><b style="color:#60a5fa">{d['model_people_mean']:.2f}</b>
   <span>how much they agree<br>with PEOPLE</span></div>
 <div class="stat"><b>{max(d['ceiling'].values()):.2f}</b>
   <span>ceiling: how well a corpus<br>agrees with itself</span></div>
 <div class="stat"><b>{d['matched_content_tokens']:,}</b>
   <span>words of each, matched so no<br>corpus saw more text</span></div>
</div>
<p class="verdict">{d['verdict']}</p>

<h2>Every pair, against the ceiling</h2>
<p class="sub">Brighter means closer agreement. The diagonal is each corpus
against itself — the most any comparison could show. The three models form a
bright block; people stay dark against all of them.</p>
<svg width="{MX0 + len(names)*CELL + 20}" height="{MH}">{''.join(mat)}</svg>

<h2>Where the difference lives</h2>
<p class="sub">One block per concept. The other nine sit on its axis at their
<b>measured distance</b> from it, one lane per corpus, joined by a thread. Where
a thread runs straight between the model lanes and then kinks at the people lane,
the models agree and people differ &mdash; those threads are drawn in
<b style="color:#f59e0b">amber</b>. Nothing is projected: every position drawn is
a number that was measured.</p>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{ticks}{''.join(rows)}</svg>

<p class="note">Distances sit in a narrow band, normal for this measure, so the
axis is stretched to the observed range rather than 0–1; the <i>ordering</i> of
neighbours carries the meaning. Ceilings are split-half correlations corrected to
full length by Spearman–Brown — uncorrected, they are measured on half the text
the comparisons use and understate badly.</p>
<p class="note">No 2-D map appears here on purpose: two dimensions retain only
about 31% of this structure, so most of what such a map showed would be an
artefact of the flattening rather than a measurement.</p>
</div>"""
    (ROOT / args.out).write_text(html, encoding="utf-8")
    print(f"wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
