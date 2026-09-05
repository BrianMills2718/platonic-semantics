# UI PROTOTYPE — IMPORTANT WARNING

`semantic_space_explorer.html` is a standalone synthetic UI prototype.

It demonstrates intended interactions such as:
- consensus view;
- one-model view;
- language residual view;
- disagreement view;
- concept inspector;
- semantic relation overlays.

All geometry and numerical scores inside that prototype are synthetic.

DO NOT:
- cite it as evidence;
- treat its clusters as discovered semantic structure;
- tune scientific conclusions to match it.

**Superseded for display purposes, 2026-09-04.** The real atlas is
`results/run_001/atlas.html`: same intent, built entirely from that run's own output
files, with no synthetic values anywhere. This file stays as the interaction sketch
and stays synthetic. Do not merge the two.

The preferred future UI should emphasize local relational structure and uncertainty rather than a single global 2-D scatterplot.

## `end_state_instrument.html` — synthetic end-state target (2026-09-04)

Also synthetic, under the same rule as the file above: it demonstrates intended
interactions and none of its groupings are discovered structure.

It exists because three attempts at a real visual all came back as "a dot plot I
could produce from any embedding in ten minutes", and the third time made the
cause obvious. Every version drew a **metric space**, and every rendering of a
metric space is points in a plane. But `docs/THEORY.md` commits to the object
being a **refinement lattice** — nested groupings with a survival threshold — not
a space at all. A 2-D scatter is the wrong form for the theory this project
holds, not a weak execution of the right one.

So this mockup draws the lattice instead: tiers of groupings that dissolve as you
raise how many independent observers must endorse them, with what dissolved named
explicitly at each level. The things a scatter cannot show are the point of it —
a grouping four observers accept and two reject, and a roster that mixes language
models with a vision encoder, human judgement and a formal ontology so a symbolic
system can check a neural one.

Gap between this and the real runs: three notions of relatedness instead of one,
six genuinely different observer types instead of six near-identical language
models, and a nested hierarchy there is something to be right or wrong about
instead of a flat word list.

