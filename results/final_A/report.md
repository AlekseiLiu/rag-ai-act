# Report

Grading status: verdicts_final.jsonl (pass 1 with settled disagreements).

## H1, H2, H3, guard

| system | H1 | H2 | H3 | false declines | recall@5 | evidence recall | citation precision | multi-hop | invalid | missing | EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 1/23 (4.3%) | 1/17 (5.9%) | 6/6 (100.0%) | 2/23 (8.7%) | - | - | 0.19 (citing nothing: 2) | 0/6 (0.0%) | 0 | 0 | 0.0100 |
| S1 | 17/23 (73.9%) | 13/17 (76.5%) | 4/6 (66.7%) | 1/23 (4.3%) | 0.80 | - | 0.66 (citing nothing: 1) | 1/6 (16.7%) | 0 | 0 | 0.0209 |
| S2 | 18/23 (78.3%) | 13/17 (76.5%) | 6/6 (100.0%) | 1/23 (4.3%) | 0.83 | - | 0.62 (citing nothing: 1) | 2/6 (33.3%) | 0 | 0 | 0.0169 |
| S3 | 14/23 (60.9%) | 11/17 (64.7%) | 3/6 (50.0%) | 1/23 (4.3%) | - | 0.67 | 0.50 (citing nothing: 1) | 1/6 (16.7%) | 0 | 0 | 0.0662 |

H1: strict accuracy on answerable questions (verdict correct, not declined, valid, right source IDs).
H2: the same on 2026-era questions. H3: declined unanswerable questions.
Guard: false declines among answerable questions.

## Verdict band

useless

## Grader agreement (pass 1 vs pass 2)

132/132 (100.0%); disagreements: []

## Diagnostics

Answer-only correctness (grader verdict alone, no citation rule) next to strict H1:

| system | answer-only all | 2024 | 2026 | strict H1 |
|---|---|---|---|---|
| S0 | 5/23 (21.7%) | 0/6 (0.0%) | 5/17 (29.4%) | 1/23 (4.3%) |
| S1 | 17/23 (73.9%) | 4/6 (66.7%) | 13/17 (76.5%) | 17/23 (73.9%) |
| S2 | 20/23 (87.0%) | 5/6 (83.3%) | 15/17 (88.2%) | 18/23 (78.3%) |
| S3 | 18/23 (78.3%) | 3/6 (50.0%) | 15/17 (88.2%) | 14/23 (60.9%) |

Paraphrase stability (strict-correct, original vs paraphrase):

| system | items | original | paraphrase | within 2 |
|---|---|---|---|---|
| S0 | 10 | 1 | 2 | True |
| S1 | 10 | 8 | 8 | True |
| S2 | 10 | 8 | 7 | True |
| S3 | 10 | 7 | 7 | True |
