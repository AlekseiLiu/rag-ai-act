# rag-ai-act

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
  the build before anything is extracted. PDFs in `data/raw/` then become legal units (one per paragraph, definition, annex point or guideline
  paragraph), each with a stable source ID and a draft flag, then chunks of at most 1,000 tokens. A coverage check
  fails the build if an article or guideline paragraph is missing or duplicated.
- **Systems.** All four share the model, the answer prompt, the chunks and the questions.
  - S0: the model alone, no sources.
  - S1: top 5 chunks by BM25.
  - S2: top 5 chunks by dense retrieval (Azure embeddings, FAISS).
  - S3: a LangGraph agent with `search`, `get_provision` (exact lookup by ID) and `applicability_date`, limited to 6 tool calls.
- **Answer contract.** Every answer is a JSON object with the answer text, the cited source IDs and a `declined` flag.
  Malformed output counts as wrong and never crashes a run.
- **Evaluation.** Answerable questions are graded blind by an LLM grader (system names and question IDs hidden), twice,
  and agreement is reported. Citations are scored mechanically against gold IDs. Unanswerable questions are scored by
  whether the system declined.
- **Measures.** H1: strict accuracy (correct, not declined, valid, right source IDs). H2: the same on questions where
  the 2026 text differs from the 2024 text. H3: unanswerable questions declined. Guard: answerable questions
  wrongly declined. A fixed verdict rule (useless, partial, good) is applied to the numbers.
- **Cost control.** A budget wrapper reserves the cost of every Azure call (token estimates with a 1.25 safety
  factor), warns at 1 EUR and stops at 3 EUR per `eval` command, keeping partial results. The cap covers the whole
  command, including an index rebuild when the saved index is missing or stale (built from other chunk texts or
  another embedding deployment); `meta.json` records index and answer spend separately and their total. A run that
  stopped early counts every missing answer as a failure and gets no verdict band ("incomplete"). What the cap
  guarantees: no new call starts once recorded spend plus open reservations would pass it, and a run stops as soon as
  recorded spend passes it, so it can be exceeded by at most the cost of the call in flight. Retries after rate-limit
  (HTTP 429) answers, which the client repeats up to ten times, are not billed by Azure.

## Results

Pending final run.

## Reproduce

Commands below call `.venv/bin/python` directly; the Makefile does the same. To type `python` instead, activate the
environment first with `source .venv/bin/activate`.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
make fetch        # verify data/raw against data/SOURCES.md (EUR-Lex files are manual downloads)
make ingest       # SHA-256 check, then data/raw -> data/processed, coverage-checked (needs pdftotext from poppler)
make check        # ruff + offline tests (no Azure needed)
```

With Azure OpenAI credentials in `.env` (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`,
`AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_EMBED_DEPLOYMENT`), a development run:

```
.venv/bin/python -m rag_ai_act eval --questions eval/dev_v0.jsonl --out results/dev_run
make chunk-compare                      # dev-set diagnostic: unit chunks vs fixed 500-token chunks
```

`eval` loads the FAISS index from `data/index/` when it matches the chunks and the embedding deployment, and
otherwise builds and saves it within the same budget (`make index` builds it on its own, with its own budget). It
writes `answers_<S>.jsonl`, `blind.jsonl`, `blind_key.json`, `meta.json` and `draft_report.md` (mechanical
metrics only). `eval/dev_v0.jsonl` is a development set used to tune prompts; the held-out set is kept apart and
used once.

### Held-out run

```
make eval HELDOUT=path/to/heldout.jsonl PARAPHRASES=path/to/paraphrases.jsonl OUT=results/heldout
make grading-export RESULTS=results/heldout GRADING_DIR=../grading_pass1     # an empty folder outside results/
# grade (see Grading), save verdicts_1.jsonl, verdicts_2.jsonl and verdicts_final.jsonl in results/heldout
make report RESULTS=results/heldout     # report.md and report.json
```

`make eval` runs `ingest` first (free, no API calls) and then a single `eval` command, so the 3 EUR cap covers the
whole run. The question files stay outside the repository; `meta.json` records only their file names and SHA-256.

## Grading

Correctness of answers to answerable questions is judged by an LLM grader that sees neither system names nor
question IDs.

1. **Export.** `.venv/bin/python -m rag_ai_act grading-export --results results/<run> --out <folder>` writes
   `<folder>/blind_stripped.jsonl`: one line per answer with only `blind_id`, `question`, `gold_answer`, `answer`
   and `declined`. Cited and gold source IDs are left out, because citations are scored mechanically. The folder
   must be empty and outside the results folder, so the grader cannot see `blind_key.json` or the answer files.
2. **Two blind passes.** The grader runs in that folder with the prompt in `eval/grading_prompt.md` and returns one
   `{"blind_id", "correct", "reason"}` line per answer; `correct` must be a JSON `true` or `false`. Two independent
   passes give `verdicts_1.jsonl` and `verdicts_2.jsonl` in the results folder; the report gives their agreement.
3. **Settlement.** Each answer the passes disagree on is decided by a person. `verdicts_final.jsonl` is pass 1 with
   those decisions applied, and is the input for H1 and H2. Without it the report uses pass 1 and says "unsettled".
4. **Report.** `make report RESULTS=results/<run>` maps verdicts back through `blind_key.json` and writes
   `report.md` and `report.json`.

## Limitations

- **Draft guidelines.** The high-risk classification guidelines are drafts and may change. Stale draft passages
  (for example 2024-era dates) are kept on purpose, and every system sees the same chunks, so none gets a cleaner corpus
  than the others.
- **S0 cannot cite documents it has never seen.** Strict accuracy therefore penalises S0 by construction; answer-only
  correctness (the grader's verdict alone, without the citation rule) is reported next to it, per system and era.
- **Small n.** The held-out set has 24 answerable and 6 unanswerable questions. One question moves a score by about
  four points; differences of one or two questions are noise.
- **LLM grading.** Correctness is judged by a language model, blind and twice, with agreement reported. It is not a
  human expert's judgement.
- **Not legal advice.** This is a research and engineering exercise.

## Credits

The AI Act and its consolidated text come from EUR-Lex (reuse under Commission Decision 2011/833/EU). The guidelines are
documents of the European Commission (CC BY 4.0). Sources and checksums are in `data/SOURCES.md`. Code is MIT licensed.
