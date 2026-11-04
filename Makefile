PY ?= .venv/bin/python
.PHONY: test lint fetch ingest index eval grading-export report reports chunk-compare check
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests
check: lint test
fetch:
	$(PY) -m rag_ai_act fetch
ingest:
	$(PY) -m rag_ai_act ingest
index: ingest
	$(PY) -m rag_ai_act index
# eval loads the saved index, or rebuilds it inside its own budget, so it does not depend on `index`.
eval: ingest
	$(PY) -m rag_ai_act eval --questions $(HELDOUT) $(if $(PARAPHRASES),--paraphrases $(PARAPHRASES)) $(if $(OUT),--out $(OUT))
grading-export:
	$(PY) -m rag_ai_act grading-export --results $(RESULTS) --out $(GRADING_DIR)
report:
	$(PY) -m rag_ai_act report --results $(RESULTS)
# Rebuild report.md and report.json in every results folder that has graded answers (offline, deterministic).
GRADED := $(patsubst %/verdicts_1.jsonl,%,$(wildcard results/*/verdicts_1.jsonl))
reports:
	@test -n "$(GRADED)" || { echo "no results/*/verdicts_1.jsonl found"; exit 1; }
	@for d in $(GRADED); do $(PY) -m rag_ai_act report --results $$d || exit 1; done
chunk-compare: ingest
	$(PY) -m rag_ai_act chunk-compare --questions eval/dev_v0.jsonl
