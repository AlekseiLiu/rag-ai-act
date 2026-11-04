# Results

Two final runs on 2026-11-04, all four systems, `gpt-4.1-mini` (temperature 0) and `text-embedding-3-small` on
Azure OpenAI. Every number below comes from the committed files in `results/final_A/` and `results/final_B/`
(`report.md`, `report.json`, `meta.json`, `verdicts_*.jsonl`). `make reports` rebuilds the reports from the verdicts,
and CI checks that the rebuild changes nothing.

| Set | Questions | Wording | Run commit | Cost |
|---|---|---|---|---|
| A (headline) | 23 answerable, 6 unanswerable, plus 10 paraphrases | the sources' own terms | `27b9df5` | 0.154 EUR |
| B (robustness) | the same 29 facts and gold answers | reworded by an agent to resemble an engineer's phrasing | `76617e2` | 0.124 EUR |

How the questions were made and checked is in [EVALUATION.md](EVALUATION.md).

## Measures

| Column | Meaning |
|---|---|
| H1 | strict accuracy on the 23 answerable questions: graded correct, not declined, valid JSON, right source IDs (multi-hop questions need both gold sources) |
| H2 | H1 on the 17 questions whose answer depends on the 2026 texts |
| H3 | unanswerable questions declined (of 6) |
| False declines | answerable questions the system declined (guard) |
| Recall@5 | S1 and S2: share of gold IDs among the 5 retrieved chunks, averaged |
| Evidence recall | S3: the same share over all IDs its tool calls returned (up to 6 calls), so not comparable with recall@5 |
| Citation precision | share of cited IDs that match a gold ID, averaged over answers citing at least one ID |
| Multi-hop | H1 on the 6 multi-hop questions |
| EUR | cost of the 29 main questions (set A's run total also includes the 10 paraphrases) |

Accepted alternative IDs (named by two or more label reviewers, see [EVALUATION.md](EVALUATION.md)) count for every
system in H1, recall and citation precision alike.

## Set A: literal questions

| System | H1 | H2 | H3 | False declines | Recall@5 | Evidence recall | Citation precision | Multi-hop | EUR |
|---|---|---|---|---|---|---|---|---|---|
| S0 model alone | 1/23 | 1/17 | 6/6 | 2/23 | - | - | 0.19 (2 cite nothing) | 0/6 | 0.010 |
| S1 BM25 | 17/23 | 13/17 | 4/6 | 1/23 | 0.80 | - | 0.66 (1 cites nothing) | 1/6 | 0.021 |
| S2 dense | **18/23** | 13/17 | **6/6** | 1/23 | 0.83 | - | 0.62 (1 cites nothing) | 2/6 | 0.017 |
| S3 agent | 14/23 | 11/17 | 3/6 | 1/23 | - | 0.67 | 0.50 (1 cites nothing) | 1/6 | 0.066 |

No invalid and no missing answers. Verdict band for the agent: **useless** (S3 is not above S1).

## Set B: reworded questions

| System | H1 | H2 | H3 | False declines | Recall@5 | Evidence recall | Citation precision | Multi-hop | EUR |
|---|---|---|---|---|---|---|---|---|---|
| S0 model alone | 1/23 | 1/17 | 6/6 | 0/23 | - | - | 0.26 (0 cite nothing) | 0/6 | 0.010 |
| S1 BM25 | 7/23 | 4/17 | 4/6 | 3/23 | 0.39 | - | 0.40 (3 cite nothing) | 1/6 | 0.022 |
| S2 dense | **12/23** | **10/17** | **6/6** | 1/23 | 0.59 | - | 0.47 (1 cites nothing) | 2/6 | 0.017 |
| S3 agent | 10/23 | 9/17 | 3/6 | 0/23 | - | 0.54 | 0.31 (0 cite nothing) | 2/6 | 0.074 |

No invalid and no missing answers. Verdict band for the agent: **partial** (above S1, below the bar for "good").

## Set A vs set B

| System | H1 A | H1 B | Change | Retrieval A -> B | Answer-only B |
|---|---|---|---|---|---|
| S0 model alone | 1/23 | 1/23 | 0 | - | 5/23 |
| S1 BM25 | 17/23 | 7/23 | -10 | recall@5 0.80 -> 0.39 | 10/23 |
| S2 dense | 18/23 | 12/23 | -6 | recall@5 0.83 -> 0.59 | 15/23 |
| S3 agent | 14/23 | 10/23 | -4 | evidence recall 0.67 -> 0.54 | 14/23 |

What this means:

1. **Retrieval is the big gain.** The model alone gets 1 of 23 right with the right source, and only 5 of 23 even
   without the citation rule. With the same model, plain retrieval gets 17 to 18 of 23 on set A.
2. **Wording decides which retriever wins.** Questions that quote the law contain rare, exact terms ("safety
   component", "fundamental rights impact assessment"). BM25 rewards rare words, so on set A it is almost as good as
   dense retrieval, cheaper and fully explainable. Set B removes those terms, as an engineer might. BM25 then
   has little to match, finds far fewer gold passages and drops from 17 to 7.
3. **Dense retrieval matches meaning**, so it holds up better (18 to 12) and is the best system on both sets.
4. **The agent loses least** (14 to 10) because it rewrites its own queries and searches again. It does not catch
   up: it answers 3 of the 6 unanswerable questions instead of declining, and cites loosely (14 right answers on set
   B, only 10 with the right source).
5. **The model alone is the floor** on both sets.

Practical reading: for an assistant over legal text, combine keyword and dense retrieval (hybrid retrieval), add
query rewriting, test on questions in the users' own words, and add an agent only once its declines and citations
are fixed.

## Answer-only correctness

The grader's verdict alone, without the citation rule. This shows that S0's low H1 is not only a citation effect.

| System | A: all | A: 2024 era | A: 2026 era | B: all | B: 2024 era | B: 2026 era |
|---|---|---|---|---|---|---|
| S0 model alone | 5/23 | 0/6 | 5/17 | 5/23 | 0/6 | 5/17 |
| S1 BM25 | 17/23 | 4/6 | 13/17 | 10/23 | 3/6 | 7/17 |
| S2 dense | 20/23 | 5/6 | 15/17 | 15/23 | 2/6 | 13/17 |
| S3 agent | 18/23 | 3/6 | 15/17 | 14/23 | 2/6 | 12/17 |

On set A, the agent's answers are often right but cite the wrong or an incomplete source: 18 right answers, 14 strict.

## Paraphrase stability (set A)

Ten set-A questions were also asked as paraphrases (other wording, clauses reordered). Strict-correct counts on the
same 10 items:

| System | Originals | Paraphrases |
|---|---|---|
| S0 model alone | 1 | 2 |
| S1 BM25 | 8 | 8 |
| S2 dense | 8 | 7 |
| S3 agent | 7 | 7 |

Every system stays within one question. These paraphrases keep most of the sources' terms, so they test wording
order more than vocabulary; set B tests vocabulary.

## Rule and predictions fixed before the runs (author's account)

The verdict rule and the predictions below were written down before either held-out run, in planning notes that
are not part of this repository, so this is the author's account rather than a public pre-registration. The public record of the freeze is the question hashes in
`eval/HELDOUT_SHA256.txt`, committed before each run.

**Verdict rule for the agent** (checked in this order; the first match wins; written as shares, shown here as counts of 23):

| Band | Condition |
|---|---|
| useless | S3 H1 at most one question above S0, or S3 H1 not above S1 |
| good | S3 H1 at least 70% (17 of 23), at least 6 questions above S0, above S1, H3 at least 5 of 6, at most 1 false decline |
| partial | everything else |

The agent was the headline system, so it had to beat BM25 to count.

| Prediction | Set A | Set B | Held? |
|---|---|---|---|
| S0 scores at most 30% strict on 2026-era questions | 1/17 | 1/17 | yes |
| On set B, BM25 loses more H1 than dense retrieval and the agent | | -10 vs -6 and -4 | yes |
| On set B, the agent loses the least H1 | | -4 | yes |
| On set B, S0 is unchanged within one question | 1/23 | 1/23 | yes |
| Agent vs chain: the agent beats dense retrieval on multi-hop by at least 2 of 6 | 1/6 vs 2/6 | 2/6 vs 2/6 | no |

| Set | S3 vs S1 | S3 vs S0 | Band |
|---|---|---|---|
| A | 14 vs 17: not above | +13 | useless |
| B | 10 vs 7: above | +9 | partial (10 of 23 is below 17) |

## Grader agreement

Graded answers: set A 132 (23 answerable questions and 10 paraphrases, four systems), set B 92 (23 questions, four
systems). Pass 1 and pass 2: Claude Opus 5.5 in fresh Claude Code sessions. Pass 3: Codex `gpt-6-sol`.

| Pair | Set A | Set B |
|---|---|---|
| Opus pass 1 vs Opus pass 2 | 132/132 | 92/92 |
| Opus pass 1 vs Codex | 129/132 | 90/92 |
| Opus pass 2 vs Codex | 129/132 | 90/92 |

The two Opus passes agreed on every answer, so their agreement says more about consistency than about accuracy.
The Codex pass is the independent check.

## Settled items

Every answer on which any grader disagreed was decided by the author against the gold answer only. Each disagreement
was the two Opus passes against Codex.

| Set | Item | System | Opus / Codex | Decision | Reason |
|---|---|---|---|---|---|
| A | H13 | S1 | wrong / correct | correct | Opens with "Yes", but the content matches the gold: measures to support AI literacy, no guaranteed level. |
| A | H14 | S3 | wrong / correct | wrong | Says that requiring human review can take the system out of high-risk, which contradicts the gold "no". |
| A | H18 | S2 | correct / wrong | correct | States the amended exception without contradicting the gold. |
| B | R12 | S2 | correct / wrong | correct | Keeps the "or" of Article 3(14); the intended-purpose test is stated correctly. |
| B | R12 | S3 | correct / wrong | wrong | Turns the "or" of Article 3(14) into "and", making failure or malfunction an extra condition. |

Two decisions changed the pass-1 verdict (A H13 S1, B R12 S3); the other three confirmed it.

## Cost and run time

| Set | S0 | S1 | S2 | S3 | Run total | Run time |
|---|---|---|---|---|---|---|
| A | 0.010 | 0.021 | 0.017 | 0.066 | 0.154 EUR (with 10 paraphrases) | 6.3 min |
| B | 0.010 | 0.022 | 0.017 | 0.074 | 0.124 EUR | 4.5 min |

The agent costs about three to four times as much as a retrieval chain per question. Both runs reused the saved index
(index cost 0 EUR) and stayed far below the 1 EUR warning.

## What would improve next

- **Agent declines.** The agent often writes "the sources do not state this" without setting the `declined` flag.
  Force the flag through structured output, then evaluate again on a fresh question set.
- **Multi-hop.** The agent rarely fetches the second provision. Add a planning step or a cross-reference hint to the tools.
- **Retrieval.** Hybrid retrieval (BM25 and dense) with query rewriting, measured on both sets.
- **Questions.** More and messier questions from several writers, in real users' phrasing, and a per-question error analysis.
- **Known parser limits.** A citation such as "Annex I point 21" parses to a point ID and never matches a
  section-level gold ID. `applicability_date` finds only provisions that Articles 111 and 113 name explicitly.
  `get_provision` on a whole article such as Article 3 returns all its units (70 definitions, about 3,500 tokens).
