#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.io as pio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="outputs/analysis")
    ap.add_argument("--out", default="outputs/semantic_relational_report.html")
    args = ap.parse_args()

    d = Path(args.analysis_dir)
    c = pd.read_csv(d / "consensus_coordinates.csv")
    r = pd.read_csv(d / "relation_convergence.csv")
    rs = pd.read_csv(d / "relation_system_scores.csv")
    align = pd.read_csv(d / "system_alignment.csv")
    ntest = pd.read_csv(d / "neighborhood_null_test.csv")
    summary = json.loads((d / "summary.json").read_text())

    fig = px.scatter(
        c, x="x", y="y", color="region", size="global_stability",
        symbol="split", hover_name="en",
        hover_data=["zh", "model_stability", "language_stability", "split"],
        title="Consensus semantic geometry — display only"
    )
    fig.update_layout(template="plotly_white", height=650)

    valid_r = r.dropna(subset=["heldout_anchor_cross_system_signature_similarity"]).copy()
    valid_r = valid_r.sort_values("heldout_anchor_cross_system_signature_similarity", ascending=False)
    fig2 = px.bar(
        valid_r, x="relation", y="heldout_anchor_cross_system_signature_similarity",
        error_y=valid_r["bootstrap_ci_hi"] - valid_r["heldout_anchor_cross_system_signature_similarity"],
        error_y_minus=valid_r["heldout_anchor_cross_system_signature_similarity"] - valid_r["bootstrap_ci_lo"],
        hover_data=["region_matched_null_mean", "region_matched_effect_over_null", "region_matched_permutation_p", "region_matched_fdr_q", "n_pairs"],
        title="Cross-system relation convergence on held-out anchors (95% bootstrap CI)"
    )
    fig2.update_layout(template="plotly_white", height=440)

    fig3 = px.scatter(
        align, x="null_mean", y="heldout_spearman_rho",
        hover_name="system_a", hover_data=["system_b", "effect_over_null", "permutation_p", "fdr_q"],
        title="Held-out system geometry alignment vs permutation null"
    )
    lim_min = min(align["null_mean"].min(), align["heldout_spearman_rho"].min(), 0)
    lim_max = max(align["null_mean"].max(), align["heldout_spearman_rho"].max(), 0.1)
    fig3.add_shape(type="line", x0=lim_min, y0=lim_min, x1=lim_max, y1=lim_max,
                   line=dict(dash="dash"))
    fig3.update_layout(template="plotly_white", height=440)

    nrow = ntest.iloc[0]
    body = f"""
    <h1>Semantic Relational Atlas — hardened pilot report</h1>
    <p><b>{summary['concepts']}</b> concepts:
    <b>{summary['selection_concepts']}</b> used for layer selection and
    <b>{summary['evaluation_concepts']}</b> held out for primary evaluation.</p>

    <div class='note'><b>Primary claim discipline:</b> layers are selected without seeing evaluation concepts.
    Headline tests use permutation nulls and FDR correction. The 2-D map is only a visualization.</div>

    <div class='cards'>
      <div><small>Held-out kNN stability</small><b>{nrow['heldout_observed']:.3f}</b></div>
      <div><small>Permutation null</small><b>{nrow['null_mean']:.3f}</b></div>
      <div><small>Effect over null</small><b>{nrow['effect_over_null']:.3f}</b></div>
      <div><small>Permutation p</small><b>{nrow['permutation_p']:.4f}</b></div>
    </div>

    {pio.to_html(fig3, full_html=False, include_plotlyjs=True)}
    {pio.to_html(fig2, full_html=False, include_plotlyjs=False)}
    {pio.to_html(fig, full_html=False, include_plotlyjs=False)}

    <h2>Selected layers</h2><pre>{json.dumps(summary['selected_layers'], indent=2)}</pre>
    <h2>Held-out system alignment</h2>{align.to_html(index=False, float_format=lambda x:f"{x:.4f}")}
    <h2>Held-out relation convergence</h2>{r.to_html(index=False, float_format=lambda x:f"{x:.4f}")}
    <h2>Within-system relation tests</h2>{rs.to_html(index=False, float_format=lambda x:f"{x:.4f}")}
    """
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
    <title>Semantic Relational Atlas report</title>
    <style>
    body{{font-family:system-ui;margin:24px auto;max-width:1200px;padding:0 18px;color:#171717}}
    .note{{padding:12px 14px;background:#f5f5f5;border-left:4px solid #666;line-height:1.45}}
    .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}}
    .cards div{{border:1px solid #ddd;border-radius:10px;padding:12px;background:#fafafa}}
    .cards small{{display:block;color:#666;margin-bottom:4px}}.cards b{{font-size:22px}}
    table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}
    @media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}}}
    </style></head><body>{body}</body></html>"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(args.out)

if __name__ == "__main__":
    main()
