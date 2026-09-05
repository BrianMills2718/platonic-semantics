#!/usr/bin/env python3
"""The Galileo map: one point per worldview, drawn flat because here that is honest.

`make_visual.py` refuses to flatten the concept space, on measurement: two
dimensions retain only ~31% of it. This figure is the level up, and the same
measurement permits it -- the space whose points are whole respondents keeps 78%
in two dimensions, so the picture is mostly the data rather than mostly the
projection. The number is printed on the figure so a reader can hold it against
the claim instead of taking the drawing on trust.

Each point is a population: a language model, or a group of people who write
about politics with a given sentiment. Distance is disagreement about how ten
political concepts relate. The circle around a point is not decoration -- it is
how far that point moves when its own corpus is split in half and placed twice,
so it is the distance sampling noise alone produces. **Two points closer together
than their circles are not measurably different**, and several here are.
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
    ap.add_argument("--data", default="results/respondent_space.json")
    ap.add_argument("--out", default="results/respondent_map.html")
    args = ap.parse_args()

    d = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    names = d["respondents"]
    P = np.array(d["coords"])
    err = d["error_radius"]
    ceil = d["ceiling"]
    D = np.array(d["disagreement"])

    W, H, PAD = 900, 700, 96
    # The extent must cover each point PLUS its error circle. Sizing on the points
    # alone clipped the largest human blobs off the canvas, which hid exactly the
    # thing those circles exist to show.
    reach = max(float(np.linalg.norm(P[i])) + err[n] for i, n in enumerate(names))
    span = max(reach, 1e-9) * 1.12
    sx = lambda x: PAD + (x / span * 0.5 + 0.5) * (W - 2 * PAD)
    sy = lambda y: PAD + (0.5 - y / span * 0.5) * (H - 2 * PAD)
    # One scale converts a disagreement radius into pixels, so circles and gaps
    # on the page are directly comparable.
    unit = (W - 2 * PAD) / (2 * span)

    y_sub_below = lambda y_name: y_name + 13
    is_model = [not n.startswith("people") for n in names]
    parts = []
    for i, n in enumerate(names):
        x, y = sx(P[i, 0]), sy(P[i, 1])
        r = max(err[n] * unit, 4)
        cls = "model" if is_model[i] else "human"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" class="err {cls}"/>')
    for i, n in enumerate(names):
        x, y = sx(P[i, 0]), sy(P[i, 1])
        cls = "model" if is_model[i] else "human"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="pt {cls}"/>')
        # Label clear of the marker, and clear of its own second line.
        up = P[i, 1] >= 0
        y_name = y - 16 if up else y + 26
        y_sub = y_name - 13 if up else y_sub_below(y_name)
        parts.append(f'<text x="{x:.1f}" y="{y_name:.1f}" class="lbl {cls}">{n}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y_sub:.1f}" class="sub">self {ceil[n]:.2f}</text>')

    n = len(names)
    pairs = sorted(((D[i, j], names[i], names[j]) for i in range(n) for j in range(i + 1, n)))
    tight = "".join(f"<tr><td>{a}</td><td>{b}</td><td class='n'>{v:.2f}</td></tr>"
                    for v, a, b in pairs[:4])
    wide = "".join(f"<tr><td>{a}</td><td>{b}</td><td class='n'>{v:.2f}</td></tr>"
                   for v, a, b in pairs[-3:])

    mm = [D[i, j] for i in range(n) for j in range(i + 1, n) if is_model[i] and is_model[j]]
    hh = [D[i, j] for i in range(n) for j in range(i + 1, n)
          if not is_model[i] and not is_model[j]]

    html = f"""<!doctype html><meta charset="utf-8">
<title>A map of worldviews</title>
<style>
 body{{margin:0;background:#0f1115;color:#e7e9ee;
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
 .wrap{{max-width:1120px;margin:0 auto;padding:34px 26px 60px}}
 h1{{font-size:27px;margin:0 0 8px;letter-spacing:-.02em}}
 .sub-p{{color:#9aa3b2;max-width:72ch;margin:0 0 18px}}
 .cols{{display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start}}
 svg{{background:#151821;border:1px solid #232838;border-radius:12px}}
 .err{{fill-opacity:.13;stroke-opacity:.42;stroke-width:1}}
 .err.model{{fill:#fb923c;stroke:#fb923c}} .err.human{{fill:#60a5fa;stroke:#60a5fa}}
 .pt.model{{fill:#fb923c}} .pt.human{{fill:#60a5fa}}
 .lbl{{font-size:13px;font-weight:700;text-anchor:middle}}
 .lbl.model{{fill:#fdba74}} .lbl.human{{fill:#93c5fd}}
 .sub{{fill:#6b7688;font-size:10.5px;text-anchor:middle}}
 table{{border-collapse:collapse;font-size:13px;margin:2px 0 16px}}
 td{{padding:4px 12px 4px 0;border-bottom:1px solid #232838}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums;color:#f59e0b}}
 h3{{font-size:13px;margin:14px 0 4px;color:#9aa3b2;text-transform:uppercase;
     letter-spacing:.06em}}
 .note{{color:#9aa3b2;font-size:13.5px;max-width:74ch;margin:18px 0 0}}
 b{{color:#e7e9ee}}
</style>
<div class="wrap">
<h1>A map of worldviews</h1>
<p class="sub-p">Woelfel's move, one level up. Each point is not a concept but a
whole population — three language models, and three groups of people writing
about politics with different sentiment. Every one of them was measured over the
same ten concepts, on the same amount of text. Distance is how much two of them
disagree about how those concepts relate.</p>

<div class="cols">
<svg width="{W}" height="{H}">{''.join(parts)}</svg>
<div>
<h3>Closest pairs</h3>
<table>{tight}</table>
<h3>Furthest apart</h3>
<table>{wide}</table>
<p class="note">Shaded circle = how far a point moves when its own corpus is
split in half and placed twice. That is sampling noise, not disagreement.
<b>Points nearer than their circles are not measurably apart.</b></p>
</div></div>

<p class="note">This map keeps <b>{d['variance_kept_2d']:.0%}</b> of the real
structure in two dimensions. The equivalent map of <i>concepts</i> keeps only
about 31%, which is why concepts are never drawn flat in this project — at that
level a flat picture would be mostly an artefact of the flattening. Here it is
mostly the measurement.</p>
<p class="note">The three models sit in a tight group (mean disagreement
<b>{np.mean(mm):.2f}</b>) while the three groups of people are scattered
(<b>{np.mean(hh):.2f}</b>). People do not form a single human position that the
models fail to reach; they disagree among themselves about as much as they
disagree with the models. Note also how much larger the human circles are: at
this corpus size the human groups are measured far less precisely than the
models, so their placements carry real uncertainty.</p>
</div>"""
    (ROOT / args.out).write_text(html, encoding="utf-8")
    print(f"wrote {ROOT / args.out}")
    print(f"\nmodel-model mean disagreement  {np.mean(mm):.3f}")
    print(f"human-human mean disagreement  {np.mean(hh):.3f}")
    print("\nclosest pairs:")
    for v, a, b in pairs[:4]:
        print(f"  {a:<18s} {b:<18s} {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
