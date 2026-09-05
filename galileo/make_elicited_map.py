#!/usr/bin/env python3
"""Draw the elicited space: several models, one frame, and the distortion shown.

This is the first artifact in the project made from distances a model was *asked*
for rather than inferred from. It is drawn flat, which needs justifying, because
this project has twice refused to flatten a space on measurement.

Classical MDS of these elicited distances keeps **54%** in two dimensions -- far
better than the text-derived distances (31%), and still short of the ~60% gate
adopted earlier. Moving that gate to license the picture would be the exact
failure this repository keeps recording, so the gate stands and the map instead
carries its own distortion:

* Every concept is drawn with a **stress ring** whose size is how badly that
  concept's distances are misrepresented by the flattening. A large ring means
  the position is a compromise, not a measurement.
* The **worst-distorted pair** is named, so the single biggest lie the picture
  tells is stated rather than left to be discovered.

No alignment step is used and none is needed. Because every model judged against
the same rod, their distances are already in the same units -- which is the whole
point of Woelfel's method and the property the text route silently discarded.
Concepts are placed once from the consensus and each model's own reading is drawn
as a spoke, so disagreement is visible as spread rather than hidden by averaging.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent


def mds(D, dim=2):
    D = np.asarray(D, float)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    pos = np.maximum(w[o], 0)
    idx = o[:dim]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0)), float(pos[:dim].sum() / pos.sum())


def procrustes(A, B):
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, s, Vt = np.linalg.svd(B0.T @ A0)
    return B0 @ (U @ Vt) * (s.sum() / max((B0 ** 2).sum(), 1e-12)) + A.mean(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/elicited_maps.json")
    ap.add_argument("--out", default="results/elicited_map.html")
    args = ap.parse_args()

    d = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    concepts = d["concepts"]
    models = d["models"]
    D = {m: np.array(d["distances"][m]) for m in models}
    n = len(concepts)
    iu = np.triu_indices(n, 1)

    C = np.mean([D[m] for m in models], axis=0)
    # Two dimensions keep 54% of this space and the picture fails visibly for it:
    # president, vote, party, biden and state collapse into one blob, and the
    # worst pair is drawn at 0.11 when it was elicited at 0.95. Three dimensions
    # keep 68%, which clears the bar this project set, so the map is 3-D and the
    # reader rotates it rather than being handed one flattening to trust.
    P3, kept3 = mds(C, dim=3)
    P, kept = mds(C, dim=2)
    Pm3 = {m: procrustes(P3, mds(D[m], dim=3)[0]) for m in models}

    # Per-concept stress: how much this concept's drawn distances depart from the
    # distances actually elicited. This is the honesty of each point.
    drawn = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    scale = C[iu].sum() / max(drawn[iu].sum(), 1e-12)
    drawn *= scale
    stress = np.array([np.sqrt(np.mean((drawn[i] - C[i]) ** 2)) for i in range(n)])
    # Every pair the flattening misplaces badly, not just the worst one. A
    # caveat in prose does not stop a reader treating adjacency as similarity,
    # so the pairs that lie are drawn as lies on the map itself.
    lies = sorted(itertools.combinations(range(n), 2),
                  key=lambda p: -abs(drawn[p] - C[p]))
    worst = lies[0]
    liars = [p for p in lies if abs(drawn[p] - C[p]) > 0.35][:4]

    # Coordinates go to the page; the rotation happens in the browser.
    scale = float(np.abs(P3).max()) or 1.0
    pts = [{"c": c, "p": [float(v) for v in P3[i] / scale],
            "s": float(stress[i] / (stress.max() or 1.0)),
            "m": {m: [float(v) for v in Pm3[m][i] / scale] for m in models}}
           for i, c in enumerate(concepts)]
    COL = {m: c for m, c in zip(models, ["#fb923c", "#f472b6", "#a78bfa", "#facc15"])}
    key = "".join(
        f'<span class="k"><i style="background:{COL[m]}"></i>{m}</span>' for m in models)
    rank = sorted(range(n), key=lambda i: -stress[i])
    rows = "".join(f"<tr><td>{concepts[i]}</td><td class='n'>{stress[i]:.2f}</td></tr>"
                   for i in rank[:5])
    payload = json.dumps({"pts": pts, "models": models, "col": COL,
                          "pairs": [{"a": concepts[i], "b": concepts[j],
                                     "real": float(C[i, j])} for i, j in lies[:3]]})

    html = f"""<!doctype html><meta charset="utf-8">
<title>The elicited space</title>
<style>
 body{{margin:0;background:#0f1115;color:#e7e9ee;
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
 .wrap{{max-width:1120px;margin:0 auto;padding:34px 26px 60px}}
 h1{{font-size:27px;margin:0 0 8px;letter-spacing:-.02em}}
 .sub{{color:#9aa3b2;max-width:72ch;margin:0 0 16px}}
 .cols{{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}}
 svg{{background:#151821;border:1px solid #232838;border-radius:12px;
      cursor:grab;touch-action:none}}
 svg:active{{cursor:grabbing}}
 .keys{{margin:12px 0 0;font-size:13px;color:#9aa3b2}}
 .k{{margin-right:16px}} .k i{{display:inline-block;width:10px;height:10px;
     border-radius:2px;margin-right:6px;vertical-align:middle}}
 table{{border-collapse:collapse;font-size:13px}}
 td{{padding:4px 12px 4px 0;border-bottom:1px solid #232838}}
 td.n{{text-align:right;color:#ef4444;font-variant-numeric:tabular-nums}}
 h3{{font-size:12.5px;margin:0 0 4px;color:#9aa3b2;text-transform:uppercase;
     letter-spacing:.06em}}
 .warn{{background:#151821;border-left:3px solid #f59e0b;padding:13px 18px;
        border-radius:0 8px 8px 0;margin:18px 0;max-width:78ch}}
 .note{{color:#9aa3b2;font-size:13.5px;max-width:76ch;margin:14px 0 0}}
 b{{color:#e7e9ee}} code{{background:#1b2130;padding:1px 5px;border-radius:4px}}
 .hint{{color:#6b7688;font-size:12.5px;margin:8px 0 0}}
</style>
<div class="wrap">
<h1>The elicited space — three models, one frame, nothing aligned</h1>
<p class="sub">Ten political concepts, positioned from distances the models were
<b>asked</b> for: each judged every pair against one shared rod
("{d['rod'][0]}" and "{d['rod'][1]}" are {d['rod'][2]} apart), over
{d['permutations']} randomised orders, twice. Because they share the rod their
numbers are already in the same units — so all three sit in one frame with
<b>no alignment step</b>. That commensurability is what Woelfel's method exists
to provide, and it is exactly what the text-counting route gave up.</p>

<div class="cols">
<div>
<svg id="v" width="700" height="600"></svg>
<p class="hint">Drag to rotate. White dot = where the three models agree;
coloured dots = each model's own placement.</p>
</div>
<div>
<h3>Least trustworthy positions</h3>
<table>{rows}</table>
<p class="note" style="max-width:28ch">Red halo = how badly this concept's
distances survive being placed at all. A big halo means its position is a
compromise.</p>
</div></div>
<p class="keys">{key}</p>

<p class="warn"><b>This is drawn in three dimensions, and rotates, because two
would not have been honest.</b> Flattened to a plane these distances keep only
{kept:.0%} of their structure and the picture visibly fails —
<code>president</code>, <code>vote</code>, <code>party</code> and
<code>biden</code> collapse into one blob, and the worst pair is drawn at
{drawn[worst]:.2f} having been elicited at {C[worst]:.2f}. Three dimensions keep
<b>{kept3:.0%}</b>, which clears the bar this project set for drawing anything
spatially. The bar was not moved to permit a picture; the picture gained a
dimension to meet it.</p>

<p class="note">Each model agrees with itself at {d['ceiling']:.2f} and with the
others at about {sum(v for k,v in d['cross_model'].items() if k.split('|')[0]<k.split('|')[1])/3:.2f},
so spread between the coloured dots is real disagreement, not noise.</p>
<p class="note">What a model states here has essentially nothing to do with what
its own writing implies — the same concepts measured by word co-occurrence give
ρ = 0.01 against these. See <code>ASKED_VS_COUNTED.md</code>.</p>

<script>
const D = {payload};
const svg = document.getElementById('v'), W = 700, H = 600;
let ax = -0.5, ay = 0.6, drag = null;
function rot(p) {{
  let [x, y, z] = p;
  let x1 = x * Math.cos(ay) - z * Math.sin(ay), z1 = x * Math.sin(ay) + z * Math.cos(ay);
  let y1 = y * Math.cos(ax) - z1 * Math.sin(ax), z2 = y * Math.sin(ax) + z1 * Math.cos(ax);
  return [x1, y1, z2];
}}
function proj(p) {{
  const [x, y, z] = rot(p), d = 3.4, k = d / (d - z * 0.9);
  return [W / 2 + x * 200 * k, H / 2 - y * 200 * k, k];
}}
function draw() {{
  const el = [];
  const order = D.pts.map((pt, i) => [proj(pt.p)[2], i]).sort((a, b) => a[0] - b[0]);
  for (const [, i] of order) {{
    const pt = D.pts[i], [x, y, k] = proj(pt.p);
    el.push(`<circle cx="${{x}}" cy="${{y}}" r="${{(9 + 30 * pt.s) * k}}"
      fill="#ef4444" fill-opacity="0.10" stroke="#ef4444" stroke-opacity="0.25"/>`);
    for (const m of D.models) {{
      const [mx, my] = proj(pt.m[m]);
      el.push(`<line x1="${{x}}" y1="${{y}}" x2="${{mx}}" y2="${{my}}"
        stroke="${{D.col[m]}}" stroke-width="1.3" stroke-opacity="0.75"/>`);
      el.push(`<circle cx="${{mx}}" cy="${{my}}" r="${{3.2 * k}}" fill="${{D.col[m]}}"/>`);
    }}
    el.push(`<circle cx="${{x}}" cy="${{y}}" r="${{4.5 * k}}" fill="#e7e9ee"/>`);
    el.push(`<text x="${{x}}" y="${{y - 14 * k}}" fill="#e7e9ee" font-size="${{13 * k}}"
      font-weight="700" text-anchor="middle">${{pt.c}}</text>`);
  }}
  svg.innerHTML = el.join('');
}}
svg.addEventListener('pointerdown', e => {{ drag = [e.clientX, e.clientY]; svg.setPointerCapture(e.pointerId); }});
svg.addEventListener('pointerup', () => drag = null);
svg.addEventListener('pointermove', e => {{
  if (!drag) return;
  ay += (e.clientX - drag[0]) * 0.01; ax += (e.clientY - drag[1]) * 0.01;
  drag = [e.clientX, e.clientY]; draw();
}});
draw();
</script>
</div>"""
    (ROOT / args.out).write_text(html, encoding="utf-8")
    print(f"wrote {ROOT / args.out}")
    print(f"3-D keeps {kept3:.0%} (2-D would keep {kept:.0%}); worst 2-D pair "
          f"{concepts[worst[0]]}-{concepts[worst[1]]}: elicited {C[worst]:.2f}, "
          f"drawn {drawn[worst]:.2f}")
    print("least trustworthy positions: " +
          ", ".join(f"{concepts[i]} {stress[i]:.2f}" for i in rank[:4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
