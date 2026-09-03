# CODING AGENT PROMPT

Take over the project in this repository.

Your goal is to empirically test whether independently trained multilingual LLMs recover shared **semantic relational geometry** across English and Chinese, and then build a data-driven atlas only if the signal survives appropriate null tests.

Read `docs/CODING_AGENT_CONTEXT.md` before changing code.

Immediate priorities:

1. Run the real pilot on:
   - Qwen/Qwen2.5-0.5B
   - bigscience/bloom-560m
   - facebook/xglm-564M
   - English + Simplified Chinese.

2. Preserve:
   - held-out concept split for layer selection/evaluation;
   - RDM-based coordinate-free comparison;
   - identity-permutation geometry null;
   - neighborhood permutation null;
   - random-target relation null;
   - region-matched relation null;
   - bootstrap uncertainty;
   - Benjamini-Hochberg FDR.

3. Add a QC layer before scientific analysis:
   - shapes;
   - concept ordering;
   - NaNs/Infs;
   - layer count;
   - hidden dimensions;
   - tokenizer statistics;
   - prompt/token inspection;
   - package/model revision metadata.

4. Run a quick analysis first, then a serious analysis with >=1000 permutations and >=2000 bootstraps.

5. Evaluate these outputs before opening the 2-D map:
   - system_alignment.csv
   - neighborhood_null_test.csv
   - relation_system_scores.csv
   - relation_convergence.csv
   - cross_language_layer_curve.csv

6. Report negative results as negative results. Do not optimize the benchmark after seeing results without explicitly labeling the work exploratory.

7. Only if robust effects exist, connect real outputs to an atlas that emphasizes:
   - local neighborhoods;
   - relation/operator explorer;
   - model/language stability;
   - layer depth;
   - disagreement;
   - uncertainty.

Do NOT:
- turn this into the separate mathematical-object atlas;
- claim PRH is proven;
- compare unrelated raw neuron coordinates;
- treat synthetic/demo files as evidence;
- make whale semantic claims;
- use ConceptNet to define latent coordinates;
- treat a 2-D projection as the primary result.

Primary scientific question:

> Which relationships among meanings remain stable when model and language change?

The most novel current metric is the coordinate-free relation signature:

    S(a,b)[j] = d(b,j) - d(a,j)

where `j` indexes shared concept anchors.

Preserve this idea and test it rigorously.
