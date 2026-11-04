# rag-ai-act

[![ci](https://github.com/AlekseiLiu/rag-ai-act/actions/workflows/ci.yml/badge.svg)](https://github.com/AlekseiLiu/rag-ai-act/actions/workflows/ci.yml)

Cited question answering over the EU AI Act (consolidated text, 2026) and the European Commission's draft guidelines
on high-risk classification, with a focus on AI in power grids. Four answer systems on Azure OpenAI are compared on a
held-out question set: no retrieval, BM25 retrieval, dense (FAISS) retrieval, and a LangGraph agent with three tools.

The author taught a university seminar on the implications of the AI Act for machine learning (2024, 2025). The Act
changed since then, and its guidelines are still drafts, so a model's memory of the law is a risky source. This project
measures whether retrieval and tool use fix that, and how often a system should decline instead of answering.

## Problem

Questions such as "Is a predictive-maintenance tool for gas pipeline monitoring high-risk under the draft guidelines?"
need the right provision, its current wording, and an honest "not in the sources" when the answer is not there. An
answer counts only if it is correct and cites the right source IDs (for example `AIA:Art6(2)` or `GLHR:para204`).

## How it works

- **Ingest.** Every PDF is first checked against its SHA-256 in `data/SOURCES.md`; a changed or missing file stops
  the build before anything is extracted. PDFs in `data/raw/` then become legal units (one per paragraph,
  definition, annex point or guideline paragraph), each with a stable source ID and a draft flag, then chunks of at
  most 1,000 tokens. A coverage check fails the build if an article or guideline paragraph is missing or duplicated.
- **Systems.** All four share the model (`gpt-4.1-mini`, temperature 0), the answer prompt, the chunks and the questions.
  - S0: the model alone, no sources.
  - S1: top 5 chunks by BM25.
  - S2: top 5 chunks by dense retrieval (`text-embedding-3-small`, FAISS).
  - S3: a LangGraph agent with `search`, `get_provision` (exact lookup by ID) and `applicability_date`, limited to 6 tool calls.
- **Answer contract.** Every answer is a JSON object with the answer text, the cited source IDs and a `declined` flag.
  Malformed output counts as wrong and never crashes a run.
- **Evaluation.** Answerable questions are graded blind by LLM graders (system names and question IDs hidden), in
  three passes, and agreement is reported. Citations are scored mechanically against gold IDs. Unanswerable questions
  are scored by whether the system declined.
- **Measures.** H1: strict accuracy (correct, not declined, valid, right source IDs). H2: the same on questions where
  the 2026 texts differ from the 2024 text. H3: unanswerable questions declined. Guard: answerable questions
  wrongly declined. A verdict rule for the agent (useless, partial, good), fixed before the runs, is applied to the numbers.
- **Cost control.** A budget wrapper reserves the cost of every Azure call (token estimates with a 1.25 safety
  factor), warns at 1 EUR and stops at 3 EUR per `eval` command, keeping partial results. The cap covers the whole
  command, including an index rebuild when the saved index is missing or stale (built from other chunk texts or
  another embedding deployment); `meta.json` records index and answer spend separately and their total. A run that
  stopped early counts every missing answer as a failure and gets no verdict band ("incomplete"). What the cap
  guarantees: no new call starts once recorded spend plus open reservations would pass it, and a run stops as soon as
  recorded spend passes it, so it can be exceeded by at most the cost of the call in flight. Retries after rate-limit
  (HTTP 429) answers, which the client repeats up to ten times, are not billed by Azure.

## Results

Two held-out sets with the same 29 facts and the same gold answers: 23 answerable (17 single-hop, 6 multi-hop) and
6 unanswerable. Set A uses the sources' own wording. Set B asks the same things reworded by an agent to resemble an engineer's phrasing (no article numbers, few legal terms).

| System | A: strict correct (H1) | A: unanswerables declined | B: strict correct (H1) | B: unanswerables declined |
|---|---|---|---|---|
| S0 model alone | 1/23 | 6/6 | 1/23 | 6/6 |
| S1 BM25 | 17/23 | 4/6 | 7/23 | 4/6 |
| S2 dense (FAISS) | **18/23** | **6/6** | **12/23** | **6/6** |
| S3 agent | 14/23 | 3/6 | 10/23 | 3/6 |
| Verdict band for the agent | useless | | partial | |

- Literal questions contain rare, exact legal terms. BM25 matches them, so on set A it is almost as good as dense retrieval.
- Reworded questions remove those terms. BM25 finds the gold passage far less often (recall@5 0.80 to 0.39) and falls from 17 to 7.
- Dense retrieval matches meaning, loses less (18 to 12) and is the best system on both sets.
- The agent loses least (14 to 10) because it rewrites its own queries, but it declines unanswerable questions less often and cites less precisely than dense retrieval.
- The model alone is the floor: 1 of 23 with the right source on either set.
- Practical reading: combine keyword and dense retrieval, add query rewriting, and test on questions in users' own words.

Grader agreement: Claude Opus vs Claude Opus 132/132 (A) and 92/92 (B); Opus vs Codex 129/132 (A) and 90/92 (B).
All tables, the predictions fixed before the runs and the settled items are in [docs/RESULTS.md](docs/RESULTS.md). How the
questions were written, checked, frozen and graded is in [docs/EVALUATION.md](docs/EVALUATION.md). A full run of
all four systems cost 0.154 EUR on set A and 0.124 EUR on set B.

## Reproduce

The Makefile calls `.venv/bin/python`. To type `python` yourself, run `source .venv/bin/activate` first.

### Level 1: free, no Azure

Needs Python 3.12 or newer (CI uses 3.14) and `pdftotext` from poppler-utils. The second line downloads tiktoken's
`cl100k_base` encoding once; after that the tests and `make ingest` make no network calls.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
make check                        # ruff + offline tests
make reports                      # rebuild report.md and report.json in every results/*/ folder from the verdicts
git diff --exit-code results/     # no output: the published numbers follow from the committed verdicts
make fetch                        # SHA-256 check of the six committed PDFs against data/SOURCES.md
make ingest                       # PDFs -> 1,249 units and 1,273 chunks, coverage-checked; needs pdftotext (poppler-utils)
```

CI runs all of Level 1 on every push, with the leak test enabled (`HELDOUT_DIR=eval/heldout`).

### Level 2: about 0.15 EUR per set, with your own Azure OpenAI resource

Deploy `gpt-4.1-mini` and `text-embedding-3-small`, then:

```
cp .env.example .env              # fill in endpoint, key, API version and your two deployment names
make eval HELDOUT=eval/heldout/heldout_v1.jsonl PARAPHRASES=eval/heldout/paraphrases_v1.jsonl OUT=results/my_set_A
make eval HELDOUT=eval/heldout/reworded_v2.jsonl OUT=results/my_set_B
```

`make eval` runs `ingest` first (free) and then one `eval` command, so the 3 EUR cap covers the whole run. The
first run also builds the FAISS index in `data/index/` (under 0.01 EUR) and reuses it afterwards. Each run writes
`answers_<S>.jsonl`, `blind.jsonl`, `blind_key.json`, `meta.json` and `draft_report.md` (mechanical metrics only:
retrieval recall, citation precision, declines, cost). `eval/dev_v0.jsonl` is the development set used to tune the
prompts; `make chunk-compare` compares unit chunks with fixed 500-token chunks on it.

### Level 3: grading

```
make grading-export RESULTS=results/my_set_A GRADING_DIR=../grading_pass1    # an empty folder outside results/
```

Run any capable LLM in that folder with the prompt in `eval/grading_prompt.md` and save its output as
`results/my_set_A/verdicts_1.jsonl`. Repeat in a fresh session for `verdicts_2.jsonl`. Decide each disagreement
yourself and write `verdicts_final.jsonl` (pass 1 with your decisions). Then run `make report RESULTS=results/my_set_A`.

LLM answers and LLM grades are not bit-for-bit reproducible, even at temperature 0, so expect small differences
from the published numbers. The published reports are fully reproducible from the committed verdicts (Level 1).

## Grading

Correctness of answers to answerable questions is judged by LLM graders that see neither system names nor
question IDs.

1. **Export.** `.venv/bin/python -m rag_ai_act grading-export --results results/<run> --out <folder>` writes
   `<folder>/blind_stripped.jsonl`: one line per answer with only `blind_id`, `question`, `gold_answer`, `answer`
   and `declined`. Cited and gold source IDs are left out, because citations are scored mechanically. The folder
   must be empty and outside the results folder, so the grader cannot see `blind_key.json` or the answer files.
2. **Blind passes.** The grader runs in that folder with the prompt in `eval/grading_prompt.md` and returns one
   `{"blind_id", "correct", "reason"}` line per answer; `correct` must be a JSON `true` or `false`. The published
   runs used three passes: two with Claude Opus 5.5 in fresh Claude Code sessions (`verdicts_1.jsonl`,
   `verdicts_2.jsonl`) and one with Codex `gpt-6-sol` (`verdicts_3.jsonl`). The report gives the agreement of
   passes 1 and 2.
3. **Settlement.** Each answer on which any grader disagrees is decided by the author. `verdicts_final.jsonl` is
   pass 1 with those decisions applied, and is the input for H1 and H2. Without it the report uses pass 1 and says
   "unsettled".
4. **Report.** `make report RESULTS=results/<run>` maps verdicts back through `blind_key.json` and writes
   `report.md` and `report.json`.

## Limitations

- **Small n.** Each set has 23 answerable and 6 unanswerable questions. One question moves a score by about four
  points; differences of one or two questions are noise.
- **One question writer.** All questions come from one drafting agent, checked by three reviewer models and the
  author. Set A reuses the sources' wording, which favours keyword search; set B was added for that reason.
- **LLM grading.** Correctness is judged by language models, blind, in three passes, with disagreements settled by
  the author. It is not a legal expert's judgement.
- **Draft guidelines.** The high-risk classification guidelines are drafts and may change. Stale draft passages
  (for example 2024-era dates) are kept on purpose, and every system sees the same chunks.
- **One chat model.** All systems use `gpt-4.1-mini`. Other models may rank the systems differently.
- **S0 cannot cite documents it has never seen.** Strict accuracy penalises S0 by construction; answer-only
  correctness (the grader's verdict alone) is reported next to it in [docs/RESULTS.md](docs/RESULTS.md).
- **Not legal advice.** This is a research and engineering exercise.

## How it was built

Built with coding agents (Claude Code and Codex) in a spec-first, test-first loop. What follows is the author's account:
the planning and review notes are not part of this repository. The public trace is the git history (commit order,
co-author trailers) and the frozen test commit.

- **Spec first.** Metrics, the verdict rule and the set-B predictions were written down before the held-out runs.
  Codex reviewed the spec in three rounds.
- **Tests first.** Codex (`gpt-6.1-sol`) wrote the test suite from a fixed test matrix, a separate Claude Opus
  session reviewed it, and it was frozen in commit `681c7be` before any implementation. Since then tests were only
  added: `git diff 681c7be HEAD -- tests/` shows new files only. The test plan is in [docs/TEST_PLAN.md](docs/TEST_PLAN.md).
- **Build.** A Claude Code orchestrator split the work into steps, gave each to a subagent (Claude Opus or
  Sonnet), and checked every hand-back against the frozen tests.
- **Cross-model reviews.** Codex reviewed the landed code (two rounds), and Claude Fable read spec, plan and code
  together. The fixes are in the git log.

The workflows are packaged as open-source skills in
[agent-workflow-skills](https://github.com/AlekseiLiu/agent-workflow-skills):
[execution-brief](https://github.com/AlekseiLiu/agent-workflow-skills/tree/main/skills/execution-brief) turned the
plan into a self-contained brief with acceptance checks per step,
[orchestrate](https://github.com/AlekseiLiu/agent-workflow-skills/tree/main/skills/orchestrate) ran the build loop,
and [review-loop](https://github.com/AlekseiLiu/agent-workflow-skills/tree/main/skills/review-loop) ran the
Codex reviews of the spec and the code.

## Credits

The AI Act and its consolidated text come from EUR-Lex (reuse under Commission Decision 2011/833/EU). The guidelines
are documents of the European Commission (CC BY 4.0). The PDFs are committed unchanged; sources, checksums and
attribution are in `data/SOURCES.md`. Code is MIT licensed.
