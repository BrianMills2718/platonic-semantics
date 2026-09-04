# REFERENCES — Shared / Platonic Semantic Space Project

These are the principal references that motivated or constrain the project.

## Core representation-convergence hypothesis

### Huh, Cheung, Wang & Isola — The Platonic Representation Hypothesis (ICML 2024)
Paper:
https://proceedings.mlr.press/v235/huh24a.html

Project page:
https://phillipi.github.io/prh/

Code:
https://github.com/minyoungg/platonic-rep

Why it matters:
- motivates comparing representational geometry rather than raw coordinates;
- argues that better models may converge in how they measure relationships among datapoints;
- includes cross-modality evidence;
- is a hypothesis/position paper and explicitly includes limitations/counterexamples.

## Multilingual internal representations

### Wendler et al. — Do Llamas Work in English? On the Latent Language of Multilingual Transformers (ACL 2024)
https://aclanthology.org/2024.acl-long.820/

Why it matters:
- studies layer-wise multilingual representations;
- proposes input-space / concept-space / output-space phases;
- finds middle-layer semantic decoding with an English bias in Llama-2.

### Brinkmann et al. — Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages (NAACL 2025)
https://aclanthology.org/2025.naacl-long.312/

Why it matters:
- finds shared feature directions for grammatical abstractions across languages;
- uses causal interventions, not only correlational similarity.

### Arnett et al. — On the Acquisition of Shared Grammatical Representations in Bilingual Language Models (ACL 2025)
https://aclanthology.org/2025.acl-long.1010/

Why it matters:
- shows crosslingual sharing can be asymmetric and weaker for less similar language pairs;
- useful counterweight to simplistic universal-representation claims.

## Important critique / constraint on PRH

### Ciernik et al. — Objective drives the consistency of representational similarity across datasets (ICML 2025)
https://proceedings.mlr.press/v267/ciernik25a.html

Code:
https://github.com/lciernik/similarity_consistency

Why it matters:
- representational similarity can be dataset dependent;
- objective function strongly affects consistency across datasets;
- motivates stimulus-set robustness tests in this project.

## The relation signature is the vector-offset method (added 2026-09-04)

With L2-normalised vectors, `d(b,j) - d(a,j) = (a_hat - b_hat) . j_hat` exactly.
The "coordinate-free relation signature" is therefore the classical vector-offset
analogy method, read out in a basis of concept anchors rather than neurons. That
reformulation is what makes it comparable across models with different hidden
sizes, which is the point -- but it also inherits everything already known about
offset analogies, and those results are the relevant prior art for the relation
half of this project. See D17 in `DECISION_LOG.md`.

### Levy & Goldberg — Linguistic Regularities in Sparse and Explicit Word Representations (CoNLL 2014)
https://aclanthology.org/W14-1618/

Why it matters:
- decomposes the offset method into similarity terms, showing what an analogy
  score is actually summing;
- introduces 3CosMul, the standard alternative to plain vector arithmetic.

### Linzen — Issues in evaluating semantic spaces using word analogies (RepEval 2016)
https://aclanthology.org/W16-2503/

Why it matters:
- shows much of offset-analogy performance is explained by plain proximity to the
  source or to the target, with no relation-specific structure required;
- the baselines it proposes are the ones this project's relation claim must beat.

### Schluter — The Word Analogy Testing Caveat (NAACL 2018)
https://aclanthology.org/N18-2039/

Why it matters:
- the standard analogy protocol excludes the query words from the answer space,
  which alone produces much of the apparent accuracy;
- directly relevant here: `relation_signature` sets `sig[a] = sig[b] = nan`, and
  that self-exclusion was supplying all the apparent pairing sensitivity of the
  retired cross-system statistic.

### Rogers, Drozd & Li — The (Too Many) Problems of Analogical Reasoning with Word Vectors (*SEM 2017)
https://aclanthology.org/S17-1017/

Why it matters:
- relation-by-relation breakdown showing offset methods succeed on a narrow band
  of relations, chiefly taxonomic and morphological, and fail elsewhere;
- predicts the shape of run 001's relation table, `IsA` included, and is the
  reason a surviving `IsA` needs a taxonomic-level-matched null rather than a
  region-matched one.

## Semantic relation probe source

### ConceptNet 5 relations
https://github.com/commonsense/conceptnet5/wiki/relations

API:
https://github.com/commonsense/conceptnet5/wiki/API

Why it matters:
- multilingual concepts connected by typed relations;
- relation names include IsA, PartOf, HasA, UsedFor, Causes, HasProperty, Antonym, SimilarTo, etc.;
- used only as an external probe/overlay, never as latent-space ground truth.

## Pilot model cards

### Qwen2.5-0.5B
https://huggingface.co/Qwen/Qwen2.5-0.5B

Relevant:
- base causal LM;
- approximately 0.5B parameters;
- multilingual support including English and Chinese.

### BLOOM-560M
https://huggingface.co/bigscience/bloom-560m

Relevant:
- BigScience multilingual causal LM;
- training corpus spans many natural and programming languages.

### XGLM-564M
https://huggingface.co/facebook/xglm-564M

Paper:
https://arxiv.org/abs/2112.10668

Relevant:
- multilingual autoregressive model;
- trained on 30 languages;
- model card reports substantial English and Chinese training data.

## Whale communication — future only

### Sharma et al. — Contextual and combinatorial structure in sperm whale vocalisations (Nature Communications 2024)
https://www.nature.com/articles/s41467-024-47221-8

Why it matters:
- establishes contextual and combinatorial structure in sperm-whale codas;
- does NOT establish semantic alignment with human language or LLM latent spaces.

The future project should compare relational structure anchored in shared observed behavioral/context variables, not assign English meanings to calls.
