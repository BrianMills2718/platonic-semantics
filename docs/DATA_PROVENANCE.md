# DATA PROVENANCE AND BENCHMARK STATUS

## Critical status

The current benchmark is a **pilot benchmark created during the design conversation**, not an established published benchmark.

### Concepts

`current_project/benchmark/concepts.csv`

SHA-256:
`9d9f51f1d6d205d11bfd88a6b12e24791a50ec2c41c4654f14c82c5afac37bdc`

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
