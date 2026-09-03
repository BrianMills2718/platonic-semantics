# Semantic Space Explorer

Open `semantic_space_explorer.html` in a browser.

This replaces the mathematical-atlas prototype. The project is about **general semantic space**, initially motivated by evidence that different LLMs and different human languages can develop partially shared representational geometry.

## The key distinction

The map is generated from neural representational structure.

External semantic systems such as WordNet, ConceptNet, Wikidata, ontologies, taxonomies, or formal mathematics can be overlaid to help interpret what the neural geometry is doing, but they are not treated as the true latent space.

## Prototype views

- **Consensus** — candidate shared semantic geometry across models/languages.
- **One model** — one model/language's geometry.
- **Language residual** — how concept positions move when the surface language changes within a model.
- **Disagreement** — concepts whose neighborhoods are least stable across systems.

The morph slider shows how a concept layout moves from consensus toward a selected model.

## Demo status

All coordinates and numerical scores in this standalone prototype are synthetic and deterministic. They are placeholders for real outputs.

## Real pipeline

A real build should:

1. define a shared concept/stimulus set;
2. create matched formulations across English, Chinese, and other languages;
3. extract layer-wise hidden representations from several independently trained models;
4. compute within-model pairwise distances or nearest-neighbor graphs;
5. align/compare relational structure rather than raw coordinates;
6. estimate consensus geometry from relationships that persist across systems;
7. quantify cross-model, cross-language, and eventually cross-modality stability;
8. project the consensus to 2D/3D only for display;
9. overlay external semantic relations as interpretive evidence, never as ground truth.

Whale communication, if added later, should enter through independently learned acoustic representations plus shared behavioral/context anchors—not by assigning human meanings to calls in advance.
