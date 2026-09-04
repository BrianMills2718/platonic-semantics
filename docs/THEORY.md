# THEORY — what "platonic semantic space" is taken to mean here

This is the deductive half of the project: what kind of object the thing is, and
what would count as representing it. It is a working position, not a result. The
inductive half — what independently trained models actually do — lives in
`RUN_001_RESULTS.md` and `FINDINGS_TO_DATE.md`.

Adopted 2026-09-04; see D18 in `DECISION_LOG.md`. It supersedes the theoretical
sections of the retired handoff briefing, whose content survives in §3 and §6.

## 1. The agenda has two ends

**Deductive.** How should a platonic semantic space be conceptualised and
represented at all? This is answerable without any model, and mostly without any
data.

**Inductive.** Independently trained models converge to some degree on
*something*. Does that convergent thing correspond to what the deductive side
describes?

The two ends are meant to meet. The value of the deductive end is not elegance —
it is that a specific account makes predictions the inductive end can check, and
those predictions are usually not "do the models agree", which is the question
that has no external referent.

## 2. The problem with the obvious approach

With ordinary concepts there is no ground-truth geometry. Nobody can say what the
correct distance between `democracy` and `gravity` is supposed to be. So a study
that compares models only to each other can measure agreement and nothing else,
and agreement alone has no shape: it is a scalar per pair, and every picture of
it is some rendering of a graph. That ceiling is structural, not a failure of
analysis or of visual design.

The deductive side exists to supply the missing referent — not as ground truth,
but as a *statement of what the models are supposed to be approximating*, sharp
enough that "does the convergent thing look like that?" is a real question.

## 3. Distance is not one thing

`fire` stands in many relations at once:

| relation | example |
| --- | --- |
| similarity | fire ↔ flame |
| causality | fire → smoke |
| contrast | fire ↔ water |
| physical | fire ↔ heat |
| function | fire → cooking |
| metaphor | fire ↔ anger |

One Euclidean distance cannot express these simultaneously, and a single 2-D
projection cannot display them. The target is therefore a **multi-relational
atlas**, not one authoritative arrangement. A global projection is a display
surface, never the object.

Consequences that this project holds to: an atlas must expose local
neighbourhoods, relation structure, cross-model and cross-language stability,
layer dependence, disagreement, and uncertainty. Local overlapping charts are
more faithful than one global map. **Disagreement is information, not noise to
be averaged away.**

## 4. The proposal: distinguishability, not position

The borrowed idea, generalised from the observation ladder in the separate
mathematics project:

> A **probe** is any question whose answer separates two things. A set of probes
> makes two things *the same point* when no probe in the set tells them apart.
> That equivalence is what an observer holding those probes can see. Adding
> probes refines the grouping. **The platonic space is the limit of that
> refinement.**

So the object is not a metric space. It is a **refinement lattice** of
distinguishability quotients, and every observer — a person, a language, a
trained model — occupies some level of it rather than seeing the whole.

Three things follow directly.

1. **"What this level forgets" is always stateable.** A coarsening is not a
   failure; it is a position in the lattice, and the information it discards can
   be named. Any honest atlas should report, for each level it shows, whether it
   *derived* that level by probing or was simply *told* it.
2. **Each kind of probe generates its own refinement.** This is §3 restated
   formally, and it is why there is no single distance function to find.
   `IsA`, `PartOf`, `UsedFor` are not merely relations to be tested inside the
   space — each is a *kind of probe*, and each induces a layer of its own.
3. **A concept's identity is its relational signature**, not a vector. Two
   concepts are the same point exactly when nothing distinguishes them; identity
   is defined by discriminability, and coordinates are at best a convenient
   encoding of it.

## 5. What this does not claim

- Not that the limit exists, is unique, or is reachable.
- Not that any observed convergence is evidence the limit is real. Models sharing
  training data, architecture and objective would converge for mundane reasons;
  see `REFERENCES.md` on Ciernik et al.
- Not that the lattice is the only defensible formalisation. It is the one this
  project commits to so that it can be wrong in a specific way.

## 6. Where mathematics sits, and why it is not this project

Mathematical objects are the region of the same lattice where the probes are
**formal and decidable** and the refinement terminates somewhere independently
checkable. `platonic-atlas-math` demonstrates exactly this: `a_p` probes added
one at a time refine 306 curves into the 93 isogeny classes, and the true
grouping is known in advance rather than inferred.

Ordinary concepts are the region where probes are informal and the limit is
unknown. Same kind of object, different epistemic access. That makes the
intuition "mathematical platonic space is a subset of a broader semantic one"
precise rather than metaphorical — the difference is what you can check, not
what kind of thing it is.

**Standing rule: this project is not the mathematics project.** Mathematics may
later serve as one semantic region, a clean validation domain, or a source of
formally specified relations. It is not the organising subject, and the two
repositories stay separate.

## 7. What the frame predicts, and what run 001 says

The lattice account makes a prediction that agreement magnitude cannot: **if each
model is a coarsening of one shared structure, their disagreements should nest.**
Groups merge as resolution drops; they do not cut across each other. Models that
found genuinely different structures produce crossing groupings.

Measured on run 001 (six systems, 15 pairs, conditional entropy against a
size-matched random-partition null; method recorded as
`lrn-20260904T182317863504Z-a321f37c26`):

| clusters | recovered toward perfect nesting | pairs beating chance |
| --- | --- | --- |
| 6 | 5.9% | 10 / 15 |
| 10 | 6.1% | 11 / 15 |
| 14 | 17.1% | 15 / 15 |
| 20 | 20.0% | 15 / 15 |
| 30 | 22.9% | 15 / 15 |

Directionally supported and weak. Every pair nests better than chance once
resolution is fine enough, but roughly two-thirds of each model's structure
remains unexplained by any other's. The honest summary is *mostly crossing, with
a consistent bias toward nesting*.

The monotone trend is the content: **nesting strengthens as you look finer.**
Three independent measurements in this run now agree on the same shape.

1. Unanimous neighbour links are within a semantic region 76% of the time at mean
   map distance 0.20, against 43% and 0.42 for links only two systems endorse.
2. Relations transfer across systems as a group-level direction (7 of 10,
   q=0.0014) and never per instance (0 of 10).
3. Grouping agreement nests locally and crosses globally, per the table above.

**These models converge on neighbourhoods and not on how neighbourhoods compose
into a whole.** That is awkward for the simple lattice picture, which predicts
the coarse divisions should be the robust ones. Two readings remain open, and
they are distinguishable by experiment:

- **(a)** There is no shared global structure at this level of description.
- **(b)** 0.5B models on bare single words sit below the resolution at which
  shared global structure appears.

The Platonic Representation Hypothesis literature predicts (b), since convergence
is claimed to rise with capability. Testing it requires larger models and is open
question 5 in `KNOWN_RISKS_AND_OPEN_QUESTIONS.md`. Until that runs, neither
reading is supported over the other, and this document should not be read as
favouring one.

## 8. How this document could be wrong

- If model disagreements are shown to cross *at every resolution* under better
  clustering methods, the coarsening account is the wrong shape for what these
  systems do.
- If nesting fails to strengthen with scale, reading (a) gains and the lattice
  stops being the useful description of neural convergence — though it may
  survive as a description of the target.
- If a single well-chosen metric predicted all the typed relations, §3 would be
  overstated and the multi-relational commitment would be unnecessary machinery.

Each of these is a way to lose, which is the point of writing it down.
