# Report

Grading status: unsettled (verdicts_1 only; no verdicts_final.jsonl).

## H1, H2, H3, guard

| system | H1 | H2 | H3 | false declines | recall@5 | evidence recall | citation precision | multi-hop | invalid | missing | EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 0/16 (0.0%) | 0/14 (0.0%) | 4/4 (100.0%) | 2/16 (12.5%) | - | - | 0.04 (citing nothing: 2) | 0/4 (0.0%) | 0 | 0 | 0.0069 |
| S1 | 10/16 (62.5%) | 9/14 (64.3%) | 4/4 (100.0%) | 1/16 (6.2%) | 0.78 | - | 0.69 (citing nothing: 1) | 0/4 (0.0%) | 0 | 0 | 0.0138 |
| S2 | 10/16 (62.5%) | 8/14 (57.1%) | 4/4 (100.0%) | 1/16 (6.2%) | 0.72 | - | 0.80 (citing nothing: 1) | 0/4 (0.0%) | 0 | 0 | 0.0111 |
| S3 | 11/16 (68.8%) | 9/14 (64.3%) | 3/4 (75.0%) | 0/16 (0.0%) | - | 0.78 | 0.59 (citing nothing: 0) | 2/4 (50.0%) | 0 | 0 | 0.0451 |

H1: strict accuracy on answerable questions (verdict correct, not declined, valid, right source IDs).
H2: the same on 2026-era questions. H3: declined unanswerable questions.
Guard: false declines among answerable questions.

## Verdict band

partial

## Grader agreement (pass 1 vs pass 2)

not available (needs verdicts_1 and verdicts_2)

## Diagnostics

Answer-only correctness (grader verdict alone, no citation rule) next to strict H1:

| system | answer-only all | 2024 | 2026 | strict H1 |
|---|---|---|---|---|
| S0 | 6/16 (37.5%) | 1/2 (50.0%) | 5/14 (35.7%) | 0/16 (0.0%) |
| S1 | 14/16 (87.5%) | 1/2 (50.0%) | 13/14 (92.9%) | 10/16 (62.5%) |
| S2 | 14/16 (87.5%) | 2/2 (100.0%) | 12/14 (85.7%) | 10/16 (62.5%) |
| S3 | 14/16 (87.5%) | 2/2 (100.0%) | 12/14 (85.7%) | 11/16 (68.8%) |

Paraphrase stability (strict-correct, original vs paraphrase):

no paraphrase items
