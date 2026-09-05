# The end point: asked spaces nest, and far more cleanly than activations

**`docs/THEORY.md` predicts that if each model is a coarsening of one shared
structure, their disagreements should *nest* — clusters merge as resolution
drops, never cut across each other. Asked spaces do.**

| | recovery toward perfect nesting |
| --- | --- |
| different models, same batch partition | 65.7% |
| **different models, different batch partitions** | **63.5%** |
| same model against itself, different partitions (ceiling) | 87.3% |
| **activations** (run 001, 9 models, 212 concepts) | **5.9 – 22.9%** |

Every resolution tested (K = 4, 6, 8, 10, 12, 15) beat its size-matched random
null, with singleton share 0–30%, so no resolution was degenerate.

Data: `results/nesting_elicited.json`, `results/nesting_control.json`.
Run: `nesting_elicited.py`, then `nesting_control.py`.

## Read it against the ceiling, not against 100%

The pipeline does not reproduce itself perfectly. One model elicited twice
through different batch partitions recovers 87.3%, and that bounds what any
cross-model comparison could show. Cross-model nesting of 63.5% is therefore
**73% of what is achievable**, not 63% of a perfect score. Run 001's activation
figures have no equivalent ceiling reported, so the comparison between the two
instruments is directional rather than exact.

## The confound that was tested and did not survive

Both spaces were first assembled through the *same* batch partition and the same
anchor pairs. Equating is imperfect — anchor spread runs 0.17–0.21 — so the same
block structure was imposed on both matrices, and the clustering could have been
recovering shared batch artefacts rather than shared semantics.

Re-eliciting one model on a different partition moves the number from 65.7% to
63.5%. The batch grouping was not doing the work.

Thresholds for withdrawing the headline were written into `nesting_control.py`
before the control data existed, so the verdict could not be tuned to the result.

## What this does and does not settle

* **Does:** on this instrument, at this scale, the models' disagreements look
  like different resolutions of one structure far more than run 001's activation
  measurements suggested. The lattice account survives a second, independent
  test — and the earlier weak result (6–23%) looks more like an instrument
  limitation than a fact about models.
* **Does not:** establish that they share *one* structure. 63.5% against an 87.3%
  ceiling leaves a real remainder that nests no better than chance.
* **Does not:** rule out shared training data, which remains the standing
  deflationary explanation for every convergence result in this project.

## Limits, and they are not small

* **One model pair.** GLM-5.2's space was lost to a stalled run and not rebuilt.
  Three models would give three pairs; this gives one.
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
