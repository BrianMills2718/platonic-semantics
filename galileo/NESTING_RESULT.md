# The end point: asked spaces nest, and far more cleanly than activations

**`docs/THEORY.md` predicts that if each model is a coarsening of one shared
structure, their disagreements should *nest* — clusters merge as resolution
drops, never cut across each other. Asked spaces do.**

Three models — `gpt-5.6-luna`, `glm-5.2`, `deepseek-v4-flash` — over 40 concepts.

| | recovery toward perfect nesting |
| --- | --- |
| different models, same batch partition | 68.1% |
| **different models, different batch partitions** | **69.4%** |
| same model against itself, different partitions (ceiling) | 87.3% |
| **activations** (run 001, 9 models, 212 concepts) | **5.9 – 22.9%** |

**All 18 comparisons beat chance** — three model pairs at each of six resolutions
(K = 4, 6, 8, 10, 12, 15) — with singleton share 0–33%, so no resolution was
degenerate.

Data: `results/nesting_elicited.json`, `results/nesting_control.json`.
Run: `nesting_elicited.py`, then `nesting_control.py`.

## Read it against the ceiling, not against 100%

The pipeline does not reproduce itself perfectly. One model elicited twice
through different batch partitions recovers 87.3%, and that bounds what any
cross-model comparison could show. Cross-model nesting of 69.4% is therefore
**80% of what is achievable**, not 69% of a perfect score. Run 001's activation
figures have no equivalent ceiling reported, so the comparison between the two
instruments is directional rather than exact.

## The confound that was tested and did not survive

Both spaces were first assembled through the *same* batch partition and the same
anchor pairs. Equating is imperfect — anchor spread runs 0.17–0.21 — so the same
block structure was imposed on both matrices, and the clustering could have been
recovering shared batch artefacts rather than shared semantics.

Re-eliciting one model on a different partition moves the number from 68.1% to
**69.4% — upward**. The batch grouping was not doing the work; if anything the
same-partition figure was very slightly the more conservative of the two.

Thresholds for withdrawing the headline were written into `nesting_control.py`
before the control data existed, so the verdict could not be tuned to the result.

## What this does and does not settle

* **Does:** on this instrument, at this scale, the models' disagreements look
  like different resolutions of one structure far more than run 001's activation
  measurements suggested. The lattice account survives a second, independent
  test — and the earlier weak result (6–23%) looks more like an instrument
  limitation than a fact about models.
* **Does not:** establish that they share *one* structure. 69.4% against an 87.3%
  ceiling leaves a real remainder that nests no better than chance.
* **Does not:** rule out shared training data, which remains the standing
  deflationary explanation for every convergence result in this project.

## Limits, and they are not small

* **Three models, one provider.** All reached through OpenRouter and all
  instruction-tuned, so nothing here separates shared structure from shared
  training data — the standing deflationary explanation across this project.
* **40 concepts against run 001's 212.** The two instruments are not compared at
  equal scale, and nesting statistics are sensitive to how many items there are.
* **One rod, one provider, three instruction-tuned models.**
* Elicitation at this scale needed common-item equating (`scale_elicit.py`),
  which is new here and carries its own error. The ceiling of 87.3% is the
  honest measure of that error's size.

## Why this is where the arm stops

The remaining capabilities — asymmetry, the ideal point, motion and the formal
inverse — extend the instrument rather than answer the question. The question was
whether the convergent structure looks like the refinement lattice. It does, more
than activations showed. Continuing past this point would be a new project, not
the completion of this one.
