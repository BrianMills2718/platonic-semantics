# Real-model execution handoff

The current ChatGPT execution environment was checked and cannot run the real-weight pilot because:

- it exposes CPU-only PyTorch;
- outbound DNS/model downloads are unavailable from the container runtime;
- `transformers`/`tokenizers` are not installed locally;
- no connected GPU/model-execution plugin is available.

No scientific result has been fabricated from the synthetic smoke-test tensors.

Use `semantic_relational_atlas_real_run.ipynb` in Google Colab with a T4 GPU or better. The notebook is self-contained: it embeds the full hardened experiment and downloads the three official pretrained model checkpoints when run.

The primary outputs to inspect are:

- `system_alignment.csv`
- `neighborhood_null_test.csv`
- `relation_system_scores.csv`
- `relation_convergence.csv`
- `cross_language_layer_curve.csv`
- `semantic_relational_report.html`
