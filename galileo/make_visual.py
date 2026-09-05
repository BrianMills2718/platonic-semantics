#!/usr/bin/env python3
"""Show the two semantic spaces exactly, without projecting them into a plane.

The obvious visual -- MDS both corpora into 2-D and draw arrows -- was built
first and then abandoned, for a measured reason. Classical MDS of these ten
concepts keeps only **31% of the structure** in two dimensions (44% in three).
Whatever such a picture shows, roughly two thirds of it is an artefact of the
projection rather than the data, and the apparent movement of a concept is then
mostly a statement about the flattening. Four earlier attempts at a visual for
this project were rejected as "basically a dot plot", and this is the likely
reason: a 2-D scatter of a space that does not fit in 2-D can only ever look
like scattered dots, because that is nearly all it contains.

So nothing is projected here. Each concept gets its own axis, and the other nine
are placed on it at their **actual measured distance** -- people on the upper
lane, the model on the lower one, joined by a line. Every number drawn is a
number that was measured. A steep line means the model moved that concept; a
flat one means the two agree about it.

The axis is stretched to the observed range rather than 0-1. PPMI cosine
distances at this vocabulary size all sit between about 0.75 and 0.91, and drawn
on a full 0-1 axis every point would collapse into one indistinguishable blob --
which would hide the entire result rather than showing it honestly.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/arm2_vs_arm3.json")
    ap.add_argument("--out", default="results/spaces.html")
    args = ap.parse_args()

    d = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    vocab = d["vocab"]
    P = np.array(d["distances"]["people"])
    M = np.array(d["distances"]["model"])
    n = len(vocab)
    iu = np.triu_indices(n, 1)
    lo, hi = min(P[iu].min(), M[iu].min()), max(P[iu].max(), M[iu].max())
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    W, LANE, ROW_H = 1080, 46, 140
    L, R = 152, W - 40

    def sx(v):
        return L + (v - lo) / (hi - lo) * (R - L)

    # Order concepts by how much the model rearranged their neighbourhood, so the
    # rows that carry the result are the ones read first.
    shift = {w: float(np.abs(P[i] - M[i]).sum()) for i, w in enumerate(vocab)}
    order = sorted(range(n), key=lambda i: -shift[vocab[i]])

    rows = []
    for slot, i in enumerate(order):
        y = 30 + slot * ROW_H
        yp, ym = y + 34, y + 34 + LANE
        seg = [f'<text x="14" y="{y + 20:.0f}" class="rowname">{vocab[i]}</text>',
               f'<text x="{L - 12}" y="{yp + 4:.0f}" class="lane">people</text>',
               f'<text x="{L - 12}" y="{ym + 4:.0f}" class="lane">model</text>',
               f'<line x1="{L}" y1="{yp}" x2="{R}" y2="{yp}" class="axis"/>',
               f'<line x1="{L}" y1="{ym}" x2="{R}" y2="{ym}" class="axis"/>']
        moves = sorted(((abs(P[i, j] - M[i, j]), j) for j in range(n) if j != i),
                       reverse=True)
        big = {j for _, j in moves[:2]}
        tagged: list[float] = []
        for j in range(n):
            if j == i:
                continue
            x1, x2 = sx(P[i, j]), sx(M[i, j])
            cls = "link big" if j in big else "link"
            seg.append(f'<line x1="{x1:.1f}" y1="{yp}" x2="{x2:.1f}" y2="{ym}" class="{cls}"/>')
            seg.append(f'<circle cx="{x1:.1f}" cy="{yp}" r="4" class="ppl"/>')
            seg.append(f'<circle cx="{x2:.1f}" cy="{ym}" r="4" class="mdl"/>')
            if j in big:
                anchor = "end" if x2 < x1 else "start"
                dx = -7 if x2 < x1 else 7
                # Two movers landing near each other overprint into an unreadable
                # smear ("copaurty"), so the second one drops to its own line.
                dy = 17 if not any(abs(x2 - px) < 95 for px in tagged) else 32
                tagged.append(x2)
                seg.append(f'<text x="{x2 + dx:.1f}" y="{ym + dy:.0f}" '
                           f'class="tag" text-anchor="{anchor}">{vocab[j]}</text>')
        rows.append("".join(seg))

    ticks = "".join(
        f'<line x1="{sx(v):.1f}" y1="18" x2="{sx(v):.1f}" y2="{30 + n * ROW_H - 26:.0f}" '
        f'class="grid"/><text x="{sx(v):.1f}" y="13" class="tick">{v:.2f}</text>'
        for v in np.linspace(lo + pad, hi - pad, 5))

    H = 30 + n * ROW_H + 10
    html = f"""<!doctype html><meta charset="utf-8">
<title>Two political semantic spaces</title>
<style>
 body{{margin:0;background:#0f1115;color:#e7e9ee;
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
 .wrap{{max-width:1140px;margin:0 auto;padding:34px 26px 56px}}
 h1{{font-size:26px;margin:0 0 8px;letter-spacing:-.02em}}
 .sub{{color:#9aa3b2;max-width:70ch;margin:0 0 10px}}
 .head{{display:flex;gap:26px;flex-wrap:wrap;margin:20px 0 26px}}
 .stat{{background:#151821;border:1px solid #232838;border-radius:10px;padding:12px 18px}}
 .stat b{{display:block;font-size:23px;font-variant-numeric:tabular-nums}}
 .stat span{{color:#9aa3b2;font-size:12.5px}}
 .verdict{{background:#151821;border-left:3px solid #f59e0b;padding:13px 18px;
           border-radius:0 8px 8px 0;margin:0 0 26px;max-width:78ch}}
 svg{{background:#151821;border:1px solid #232838;border-radius:12px}}
 .axis{{stroke:#2b3242;stroke-width:1}}
 .grid{{stroke:#1d2330;stroke-width:1}}
 .tick{{fill:#6b7688;font-size:11px;text-anchor:middle}}
 .link{{stroke:#475569;stroke-width:1.3}}
 .link.big{{stroke:#f59e0b;stroke-width:2.2}}
 .ppl{{fill:#60a5fa}} .mdl{{fill:#fb923c}}
 .rowname{{fill:#e7e9ee;font-size:15.5px;font-weight:700}}
 .lane{{fill:#6b7688;font-size:10.5px;text-anchor:end;text-transform:uppercase;
        letter-spacing:.06em}}
 .tag{{fill:#f59e0b;font-size:11px}}
 .note{{color:#9aa3b2;font-size:13.5px;max-width:76ch;margin:22px 0 0}}
 code{{background:#1b2130;padding:1px 5px;border-radius:4px;font-size:12.5px}}
</style>
<div class="wrap">
<h1>Do people and a language model organise politics the same way?</h1>
<p class="sub">Ten political concepts, positioned by how they are actually
<i>used</i> — in {d['n_posts']['people']:,} posts written by people, and
{d['n_posts']['model']:,} written by a language model asked to post about the same
topics. One identical instrument reads both, on the same amount of text, so the
only thing that differs is who wrote it.</p>

<div class="head">
 <div class="stat"><b style="color:#60a5fa">{d['ceiling']:.2f}</b>
   <span>how well each corpus agrees with ITSELF<br>— the most any comparison could show</span></div>
 <div class="stat"><b style="color:#fb923c">{d['cross_arm_rho']:.2f}</b>
   <span>how well people and the model agree<br>with each other</span></div>
 <div class="stat"><b>{d['matched_content_tokens']:,}</b>
   <span>words of each, matched<br>so neither side saw more text</span></div>
</div>
<p class="verdict">{d['verdict']}</p>

<p class="sub">Each block below is one concept. The other nine are placed on its
axis at their <b>measured distance</b> from it — people on the upper lane, the
model on the lower. A steep line means the model moved that concept; a flat line
means they agree. Nothing is projected or flattened: every position drawn is a
number that was measured. Rows are ordered by how much the model rearranged that
concept's neighbourhood, and its two largest movers are named.</p>

<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{ticks}{''.join(rows)}</svg>

<p class="note">Distances are crowded into a narrow band (0.75–0.91), which is
normal for this measure and is why the axis is stretched to that range rather
than 0–1. It also means the <i>ordering</i> of neighbours carries the meaning,
not the absolute numbers.</p>
<p class="note">The obvious alternative — flatten both spaces to a 2-D map and
draw arrows — was built first and rejected on evidence: two dimensions retain
only <b>31%</b> of this structure, so most of what such a map shows would be an
artefact of the flattening rather than the measurement.</p>
</div>"""
    out = ROOT / args.out
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    print("\nconcepts whose neighbourhood the model rearranged most:")
    for i in order[:5]:
        j = max((k for k in range(n) if k != i), key=lambda k: abs(P[i, k] - M[i, k]))
        who = "model holds them closer" if M[i, j] < P[i, j] else "people hold them closer"
        print(f"  {vocab[i]:<11s} total shift {shift[vocab[i]]:.2f}"
              f"   biggest: {vocab[j]} ({who})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
