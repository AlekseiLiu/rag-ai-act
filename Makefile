PY ?= .venv/bin/python
.PHONY: test lint fetch ingest index eval report chunk-compare check
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
eval: index
	$(PY) -m rag_ai_act eval --questions $(HELDOUT) $(if $(PARAPHRASES),--paraphrases $(PARAPHRASES)) $(if $(OUT),--out $(OUT))
report:
	$(PY) -m rag_ai_act report --results $(RESULTS)
chunk-compare: ingest
	$(PY) -m rag_ai_act chunk-compare --questions eval/dev_v0.jsonl
