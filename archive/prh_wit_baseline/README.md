# Empirical Platonic Map Builder

This project generates an **actual empirical map of shared representational geometry** from real neural-network activations.

It is based on the public precomputed WIT-1024 features released with:

> Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola.  
> *The Platonic Representation Hypothesis.* ICML 2024.

Official paper/project:
- https://proceedings.mlr.press/v235/huh24a.html
- https://phillipi.github.io/prh/
- https://github.com/minyoungg/platonic-rep

## What the map means

Each point is one stimulus seen by every included model.

The script **does not concatenate hidden states** from different networks. That would be meaningless because the coordinate systems differ.

Instead it:

1. Calculates a cosine-distance matrix among the same stimuli inside each model/layer.
2. Chooses a layer in each model that best agrees with the others in ranked pairwise geometry.
3. Rank-normalizes each selected distance matrix.
4. Averages the distance geometry across models.
5. Projects that consensus distance matrix to two dimensions using classical MDS.
6. Measures how consistently different models agree on each point's local nearest neighbors.

So the output is a candidate **shared geometry**, not a claim that every model literally contains the same vectors.

## Install

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Generate the first map

```bash
python platonic_map.py
```

Defaults use:
- BLOOMZ-560M (language)
- BLOOMZ-1.1B (language)
- ImageNet-21K ViT-Tiny (vision)
- DINOv2-Small (vision)
- CLIP ViT-Base (vision)

and 384 of the 1,024 shared WIT stimuli. The activations are downloaded from the PRH authors' MIT-hosted feature archive.

Outputs:

```text
platonic_map_output/
  platonic_map.html
  platonic_map_coordinates.csv
  platonic_map_metadata.json
```

Open `platonic_map.html` in a browser.

## More demanding map

If you have enough RAM/bandwidth:

```bash
python platonic_map.py \
  --models bloom_560m bloom_1b1 bloom_1b7 openllama_3b dinov2_s clip_b clip_l \
  --items 768 \
  --clusters 14
```

The raw feature downloads can be hundreds of MB. They are cached locally after the first run.

## How to interpret it

- **Distance:** consensus representational dissimilarity.
- **Point size:** agreement among models about that item's local neighbors.
- **Color:** unsupervised cluster; it is descriptive, not a known semantic category.
- **Hover text:** WIT caption when the `wit_1024` Hugging Face revision is available.
- **Model table:** which layer was selected and how well its geometry correlates with the consensus.

A 2D projection is only a view. The high-dimensional consensus distance matrix is the scientifically important object.

## The multilingual experiment I would do next

Do **not** just put English words and Chinese words into one embedding model and call the result universal.

Use parallel stimuli describing the same events/concepts in, for example:

- English
- Mandarin Chinese
- Spanish
- Arabic
- Japanese

For each language and each multilingual model:

1. extract hidden activations layer-by-layer;
2. pool only non-padding tokens;
3. build an RDM over the same underlying stimuli;
4. compare RDMs across languages and models;
5. add language-specific RDMs to the consensus only after measuring how much geometry survives.

The key test is whether **relationships among stimuli** survive language changes, not whether raw vectors are numerically similar.

Good model families for a modern follow-up would include several independently trained multilingual/open-weight families rather than several sizes from one family.

## Why whales should be a separate phase

Whale communication is the scientifically difficult part.

You cannot responsibly insert a sperm-whale coda into a human-language semantic map merely because both have learned embeddings. There is no established point-to-point semantic dictionary.

A defensible whale experiment needs anchors such as:

- the same individual or social group;
- behavioral context;
- conversational turn position;
- dive/foraging/social state;
- repeated coda types or learned acoustic units;
- synchronized environmental observations.

Then compare **relational geometry** between whale-acoustic embeddings and representations of those shared contexts. If stable alignment appears without hand-assigning human meanings, *that* would be genuinely interesting evidence.

## Important caveat

The Platonic Representation Hypothesis is a hypothesis, not a settled result. Subsequent work has shown that apparent convergence depends on metric, objective, and dataset. Treat this map as an analysis instrument for testing convergence, not a picture of a proven universal ontology.
