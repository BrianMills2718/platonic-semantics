# EXPERIMENT 0 FREEZE RECORD

Created before the first real pretrained-model output, and re-frozen on
2026-09-03 after a defect was found in the relation null and corrected. No real
run had occurred, so nothing confirmatory was invalidated: this record freezes
the corrected artifacts, and the hashes below supersede the earlier v2-handoff
record. The change rule at the bottom applies from here on.

The superseded artifacts, and why they changed, are in `FINDINGS_TO_DATE.md`
and `DECISION_LOG.md` (D16).

## Benchmark SHA-256

concepts.csv (212 concepts):
`9d9f51f1d6d205d11bfd88a6b12e24791a50ec2c41c4654f14c82c5afac37bdc`

relations.csv (145 pairs, every relation type all-distinct targets):
`63510a698d04c1cd76e6d4c8f3cc347219de2c892d73913ebe73ae0b8cc570f1`

## Core code SHA-256

extract_representations.py:
`cdf230a07b5783b762a5087be280f90fac542ad1f5346acf2007baf5abf09ccd`

analyze_semantic_geometry.py:
`64b32bf441e0830809a53d1969f8564a478837dc5b661995d15a1f70a109fc51`

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
