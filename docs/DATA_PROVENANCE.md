# DATA PROVENANCE AND BENCHMARK STATUS

## Critical status

The current benchmark is a **pilot benchmark created during the design conversation**, not an established published benchmark.

### Concepts

`current_project/benchmark/concepts.csv`

SHA-256:
`73dbb915503fcb5e8c6e6c0bb5bba262c2d5785ee9e1a5a8699abcd74838b8db`

212 concepts with:
- stable pilot ID;
- English term;
- Simplified Chinese term;
- broad semantic region.

### Relations

`current_project/benchmark/relations.csv`

SHA-256:
`63510a698d04c1cd76e6d4c8f3cc347219de2c892d73913ebe73ae0b8cc570f1`

145 directed probe pairs across 10 relation types, **every type with
all-distinct targets**. This is a hard design constraint, enforced by
`tests/test_null_calibration.py`, not a stylistic preference: with reused
targets the correct multiplicity-preserving null converges on the observed data
and the relation test has no power at all. 38 concepts (mostly superordinates
such as `tool`, `metal`, `insect`, `emotion`) were added to the original 174
specifically to make distinct-target probes constructible.

## Provenance warning

The inventory, Chinese lexical equivalents, broad regions, and relation examples were assistant-curated for a runnable pilot.

Therefore:
- Chinese terms have **not had independent bilingual review**;
- senses are not formally disambiguated;
- broad semantic regions are hand labels;
- relation examples are curated rather than population-sampled;
- the 38 concepts added on 2026-09-03 were assistant-curated on the same basis
  as the original 174 and have had no more review than they did;
- `study` was changed from 学习 to 研究 on 2026-09-03 because `learn` also carried
  学习, making the two concepts byte-identical in every Chinese system. 研究 is
  closer to "research" than to "study" and is exactly the kind of choice that
  needs the bilingual review this benchmark has never had;
- six Chinese terms contain an `<unk>` token in XGLM only — 哺乳动物, 鲸, 锤子,
  钥匙, 烹饪, 谬误 — 2.8% of the benchmark. Of those, only 鲸 (whale) and 烹饪
  (cook) reduce *entirely* to `<unk>` and are therefore mutually
  indistinguishable; the other four retain resolved content tokens. Corrected
  2026-09-04 from an earlier wording that called all six indistinguishable.
  Qwen and BLOOM cover the set fully.
  `scripts/qc_representations.py --tokenizers` re-measures this;
- the benchmark is not simply a ConceptNet export;
- symmetric relations are still stored as directed rows;
- inverse families such as PartOf/HasA introduce dependency.

Any scientific report must disclose this.

## Freeze rule for Experiment 0

Do not edit concepts/translations/relation pairs after seeing real headline results and then treat the revised run as if it were the original confirmatory test.

If an error is found:
1. preserve the original output;
2. log the problem;
3. create a new benchmark version;
4. rerun;
5. label the new analysis corrected/exploratory/replication as appropriate.

## Needed before stronger claims

- bilingual review;
- canonical sense IDs;
- POS metadata;
- lexical-frequency metadata;
- tokenization statistics;
- contextualized examples;
- relation provenance/confidence;
- more relation examples;
- matched negative/control pairs;
- independent benchmark replication.
