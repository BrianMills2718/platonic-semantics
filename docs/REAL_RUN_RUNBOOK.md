# REAL RUN RUNBOOK

## Recommended route: a local GPU

Run 001 settled this. The handoff said the run needed Colab; it did not. Three
0.5B models over 212 short prompts is about two minutes of extraction on a local
NVIDIA T600 (4 GB) in float32, one model loaded at a time.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pilot.py --device cuda --permutations 1000 --bootstrap 2000
```

Use `--device mps` on Apple Silicon, or omit `--device` to autodetect. Always
run `scripts/qc_representations.py` on the extracted tensors before believing any
number: run 001's two real defects were both invisible in the code and visible in
the tensors.

## Fallback route: Colab

Open:

`notebooks/semantic_relational_atlas_real_run.ipynb`

Use:
- T4 GPU or better;
- internet access;
- sufficient disk for three ~0.5B model checkpoints.

Run all cells.

The notebook:
- installs dependencies;
- contains the experiment;
- downloads model weights;
- extracts representations;
- runs statistics;
- produces an HTML report;
- bundles outputs.

## Quick diagnostic

For command-line project use:

```bash
python run_pilot.py --device cuda --permutations 100 --bootstrap 200
```

## Serious pilot

```bash
python run_pilot.py --device cuda --permutations 1000 --bootstrap 2000
```

## Before trusting results

Create a QC report covering:
- model revisions;
- package versions;
- GPU;
- concept counts;
- EN/ZH ordering equality;
- representation shape per system;
- finite values;
- tokenizer token lengths;
- special-token handling;
- extraction prompt examples.

## Primary result order

1. system_alignment.csv
2. neighborhood_null_test.csv
3. relation_convergence.csv
4. relation_system_scores.csv
5. cross_language_layer_curve.csv
6. consensus_coordinates.csv
7. HTML report

The 2-D projection comes last.
