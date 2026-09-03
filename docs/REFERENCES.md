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
