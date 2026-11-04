# Report

Grading status: verdicts_final.jsonl (pass 1 with settled disagreements).

## H1, H2, H3, guard

| system | H1 | H2 | H3 | false declines | recall@5 | evidence recall | citation precision | multi-hop | invalid | missing | EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 1/23 (4.3%) | 1/17 (5.9%) | 6/6 (100.0%) | 0/23 (0.0%) | - | - | 0.26 (citing nothing: 0) | 0/6 (0.0%) | 0 | 0 | 0.0104 |
| S1 | 7/23 (30.4%) | 4/17 (23.5%) | 4/6 (66.7%) | 3/23 (13.0%) | 0.39 | - | 0.40 (citing nothing: 3) | 1/6 (16.7%) | 0 | 0 | 0.0222 |
| S2 | 12/23 (52.2%) | 10/17 (58.8%) | 6/6 (100.0%) | 1/23 (4.3%) | 0.59 | - | 0.47 (citing nothing: 1) | 2/6 (33.3%) | 0 | 0 | 0.0174 |
| S3 | 10/23 (43.5%) | 9/17 (52.9%) | 3/6 (50.0%) | 0/23 (0.0%) | - | 0.54 | 0.31 (citing nothing: 0) | 2/6 (33.3%) | 0 | 0 | 0.0743 |

H1: strict accuracy on answerable questions (verdict correct, not declined, valid, right source IDs).
H2: the same on 2026-era questions. H3: declined unanswerable questions.
Guard: false declines among answerable questions.

## Verdict band

partial

## Grader agreement (pass 1 vs pass 2)

92/92 (100.0%); disagreements: []

## Diagnostics

Answer-only correctness (grader verdict alone, no citation rule) next to strict H1:

| system | answer-only all | 2024 | 2026 | strict H1 |
|---|---|---|---|---|
| S0 | 5/23 (21.7%) | 0/6 (0.0%) | 5/17 (29.4%) | 1/23 (4.3%) |
| S1 | 10/23 (43.5%) | 3/6 (50.0%) | 7/17 (41.2%) | 7/23 (30.4%) |
| S2 | 15/23 (65.2%) | 2/6 (33.3%) | 13/17 (76.5%) | 12/23 (52.2%) |
| S3 | 14/23 (60.9%) | 2/6 (33.3%) | 12/17 (70.6%) | 10/23 (43.5%) |

Paraphrase stability (strict-correct, original vs paraphrase):

no paraphrase items
