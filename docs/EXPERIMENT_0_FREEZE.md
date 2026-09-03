# EXPERIMENT 0 FREEZE RECORD

Created before the first real pretrained-model output, and re-frozen on
2026-09-03 twice: first after a defect was found in the relation null, then
again after the first real extraction exposed two more (bloom-560m returning NaN
in float16, and `learn`/`study` sharing one Chinese term). Both are recorded in
`FINDINGS_TO_DATE.md`. No real
run had occurred, so nothing confirmatory was invalidated: this record freezes
the corrected artifacts, and the hashes below supersede the earlier v2-handoff
record. The change rule at the bottom applies from here on.

The superseded artifacts, and why they changed, are in `FINDINGS_TO_DATE.md`
and `DECISION_LOG.md` (D16).

## Benchmark SHA-256

concepts.csv (212 concepts):
`73dbb915503fcb5e8c6e6c0bb5bba262c2d5785ee9e1a5a8699abcd74838b8db`

relations.csv (145 pairs, every relation type all-distinct targets):
`63510a698d04c1cd76e6d4c8f3cc347219de2c892d73913ebe73ae0b8cc570f1`

## Core code SHA-256

extract_representations.py:
`3d7b1bebfca2dbdb2d387fd0b1cb704946b68cadd8d742c21f429aa821845898`

analyze_semantic_geometry.py:
`9caab82176f223607d433c939954d6b527c88025c3a498329aac90396a5e92c5`

## Frozen pilot choices

Models:
- Qwen/Qwen2.5-0.5B
- bigscience/bloom-560m
- facebook/xglm-564M

Languages:
- en
- zh

Primary prompt:
- bare lexical term

Primary analysis:
- cosine RDM;
- stratified ~60/40 selection/evaluation concept split;
- kNN k=10;
- relation signature `S(a,b)[j] = d(b,j) - d(a,j)`;
- identity permutation geometry null;
- relabeled-system neighborhood null;
- random-target relation null (multiplicity-preserving);
- region-matched relation null (multiplicity-preserving);
- permuted-pairing relation null, the strictest within-system baseline;
- bootstrap uncertainty;
- Benjamini-Hochberg FDR.

## Change rule

If these benchmark or primary analysis choices change after seeing real headline outputs:
- preserve the original run;
- create a new experiment version;
- record why it changed;
- label the new analysis exploratory/corrected/replication as appropriate.
