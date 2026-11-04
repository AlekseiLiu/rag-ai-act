# Evaluation

How the questions were written and checked, how they were kept from the systems' builders, and how the answers were
graded. The numbers are in [RESULTS.md](RESULTS.md).

## Question files

| File | Items | Use |
|---|---|---|
| `eval/dev_v0.jsonl` | 20 (16 answerable, 4 unanswerable) | prompt tuning, two dev runs |
| `eval/heldout/heldout_v1.jsonl` | 29 (17 single-hop, 6 multi-hop, 6 unanswerable) | set A, the headline run |
| `eval/heldout/paraphrases_v1.jsonl` | 10 paraphrases of set-A questions | paraphrase stability in the set-A run |
| `eval/heldout/reworded_v2.jsonl` | 29, one per set-A question | set B, the robustness run |

Each record has `qid`, `question`, `type` (`single`, `multi` or `unanswerable`), `era`, `gold_answer`, `gold_ids`
and verbatim `evidence` quotes. Some records also have `also_ok_ids` (accepted alternative IDs) and `stale_answer`
(what an outdated reading of the law would say). Paraphrases link to their original with `paraphrase_of`, reworded
questions with `reworded_of`.

## How the questions were written

This section and the label check below are the author's account; the working notes are not in this repository.

One Claude Opus agent wrote all questions in one session, before any answer system existed. It saw only the corpus
text and, to write "what changed" questions, the original 2024 Act. It had no web access and called no model.

The brief was fixed in advance: 30 held-out questions (18 single-hop, 6 multi-hop, 6 unanswerable), at least half
needing the 2026 texts, at least half about critical infrastructure and power grids, 10 paraphrases, and 20 dev
questions with no overlap with the held-out ones. Answers had to be short and checkable: yes or no with a reason, a
date, or a provision.

The method is the usual one for corpus-grounded test sets: pick a passage, write a question it answers, write the
gold answer from the passage, and record the passage's ID.

- **Single-hop**: one provision answers the question.
- **Multi-hop**: two provisions are needed, and a strict-correct answer must cite both.
- **Unanswerable**: the sources do not contain the answer; the right response is to decline.
- **Era**: `2024` if the answer already follows from the 2024 Act as published, `2026` if it depends on text the
  model is unlikely to know: the 2026 consolidated text or the Commission guidelines. Set A has 6 answerable
  questions of era 2024 and 17 of era 2026.
- **Gold IDs** name the paragraph, definition, annex point or guideline paragraph in the project's ID scheme (for
  example `AIA:Art113(3)`, `GLHR:para204`).
- **Evidence quotes** are verbatim. Each was checked by script to occur exactly once in the extracted source text,
  and to lie inside the unit its gold ID names.

## Label check

Three reviewer models (Claude Opus 5.5, Claude Fable 5.1 and Codex `gpt-6-astra`) checked all 40 held-out items
(30 questions and 10 paraphrases) against the full source text and the parsed units, independently of each other.

- **Unanimous items were accepted.** All three reviewers marked 24 of the 40 items as correct.
- **Disputed items were decided by the author.** For the other 16 items, the author decided each one. Most fixes
  added a condition to the question that the source requires for the gold answer (for example that a data-centre operator
  is designated as a critical entity), or brought a paraphrase back in line with its original. One gold answer was
  completed, and two era tags were corrected.
- **One item was dropped.** Its answer stood only in a guideline footnote, and ingest removes footnotes, so no
  system could retrieve it. This left 29 questions: 23 answerable and 6 unanswerable.
- **Alternative IDs.** When two or more reviewers named another passage that also supports the gold answer, it was
  accepted as `also_ok_ids` (four ID pairs, used by eight items). Alternative IDs count for every system alike, in
  H1, recall and citation precision.

## Set B: reworded questions

Set A reuses the sources' wording, because each question was written from a passage. That favours keyword search.
Before either held-out run, a BM25 rank check over the real chunks showed it: for most set-A questions the gold
passage was BM25's top-ranked chunk (median rank 1).

Set B was added to test this. A fresh Claude Opus agent rewrote every set-A question to resemble how an engineer might ask it:
same facts, same gold answers and gold IDs, but without article numbers and with few of the sources' own terms. Two
reviewers (Claude Opus and Codex) checked that each rewrite still asks the same thing. Where a rewrite changed the
meaning, for example by dropping a condition or narrowing an "or" to an "and", the author applied the reviewers'
corrected wording (16 of 29 items). The predictions for set B were written down before either held-out run; they
are listed with their outcomes in [RESULTS.md](RESULTS.md).

## Freezing

- **Set A** (questions and paraphrases) was frozen after the label check. Its SHA-256 hashes were committed in
  `eval/HELDOUT_SHA256.txt` (commit `27b9df5`) before the final run, and `results/final_A/meta.json` records that
  same commit and the same file hashes.
- **Set B** was frozen before its run. Its hash was committed in `aa5e0d6`, and the set-B run started from a later
  commit (`76617e2`).
- **Public copies.** The question files in `eval/heldout/` were published after the set-A run. They omit only the
  drafting notes; questions, types, eras, gold answers, gold IDs, alternative IDs and evidence quotes are identical
  to the frozen files. `eval/HELDOUT_SHA256.txt` lists the hashes of both the frozen files and the public copies.
- **Each set was run once.** No prompt or code changed between the freeze and the run.

## Kept apart from the build

- The coding agents that built the systems never saw the held-out questions; they worked against the dev set and
  the test suite.
- A test (`test_no_evaluation_answers_embedded_in_code` in `tests/test_grading.py`) fails if any held-out question
  ID, question or gold answer appears in `src/` or `tests/`. It runs when `HELDOUT_DIR` points to the question
  files, for example `HELDOUT_DIR=eval/heldout`.
- The dev set was used to tune the answer prompts in two dev runs (`results/dev_2026-11-04/` and
  `results/dev_2026-11-04_run2/`, one grading pass each). After the second dev run the systems were frozen. Scores
  did not change between the two dev runs (S0 0/16, S1 10/16, S2 10/16, S3 11/16 strict), so more tuning would
  have chased noise.

## Grading and settlement

- **Mechanical parts.** Citations, retrieval recall, declines, validity and cost are computed by script from the
  answer files.
- **Blind correctness.** `grading-export` writes a stripped file with only `blind_id`, `question`, `gold_answer`,
  `answer` and `declined`, shuffled with a fixed seed, into an empty folder outside the results. The grader sees no
  system names, no question IDs and no citations. The prompt is `eval/grading_prompt.md`: judge only against the
  gold answer, not against one's own knowledge of the law.
- **Three passes.** Two with Claude Opus 5.5 in fresh Claude Code sessions and one with Codex `gpt-6-sol`. The
  answer models (`gpt-4.1-mini`) never grade themselves.
- **Settlement.** Every answer on which any grader disagreed was decided by the author against the gold answer. Set
  A had 3 such answers, set B had 2. `verdicts_final.jsonl` is pass 1 with these decisions, and H1 and H2 come from it.
- **Report.** `make report` maps the verdicts back through `blind_key.json`. The report code has a test on a small
  hand-computed example, and `make reports` rebuilds every published report from the committed verdicts.

The specification and the code were reviewed by a different model than the one that wrote them: Codex reviewed
both, with a cap on rounds and a recheck of each fix, using the
[review-loop](https://github.com/AlekseiLiu/agent-workflow-skills/tree/main/skills/review-loop) skill. A separate
Claude session reviewed the tests, which Codex had written. These reviews closed loopholes such as a multi-hop
answer passing with only one of its two sources, a grade written as the text "false" counting as correct, and no
test catching an agent that keeps calling the model after its tool limit.

## Known weaknesses

- **One question writer.** One agent wrote all questions, so they share one style: clear facts in one or two
  paragraphs, the easy case for retrieval. Three reviewer models and the author checked the labels, not the style.
- **Lexical overlap of set A.** Measured before the run (see above). It flatters keyword search on set A, which is
  why set B exists. Set B was written by the same kind of agent from set A, so it is still not real user phrasing.
- **Gold IDs list the passage the writer used.** Another passage may also support the answer. Alternative IDs named
  by two or more reviewers are accepted; others are still scored as wrong citations.
- **Small n.** 23 answerable questions per set: one question is about four points, and differences of one or two
  questions are noise.
- **LLM grading.** The graders are language models. The two Opus passes agreed on every answer, which shows
  consistency, not correctness; the Codex pass disagreed on 3 of 132 and 2 of 92 answers, and the author decided
  those. No human expert graded all answers.
- **Draft guidelines.** The high-risk guidelines are drafts and may change; some of their passages are already
  stale after the 2026 amendments and are kept in the corpus on purpose.
