"""Command line: python -m rag_ai_act <command> (Makefile targets wrap these).

  fetch                       verify data/raw against data/SOURCES.md sha256; download what can be scripted
  ingest                      data/raw -> data/processed/{units,chunks}.jsonl (coverage-checked)
  index                       build FAISS index in data/index/ (embeddings, budget-capped)
  eval --questions P [--paraphrases P2] [--systems S0,S1,S2,S3] [--out results/<date>_<tag>]
                              P2 records: same keys as questions plus "paraphrase_of": <qid of the original>;
                              their answers go into the same answers_<S>.jsonl (paraphrase qids) and into the
                              blind export; report.md uses them only for paraphrase stability
                              run systems, write answers_<S>.jsonl, blind.jsonl, blind_key.json, meta.json,
                              draft_report.md (mechanical metrics only); stops at the budget, keeping results.
                              meta.json records the questions file by basename and sha256 only (never a path)
  report --results DIR        H1/H2 come from verdicts_final.jsonl (grading pass 1 with the author's settled
                              disagreements); if absent, from verdicts_1.jsonl and the report says "unsettled".
                              Agreement of verdicts_1 vs verdicts_2 reported when both exist. report.md: H1, H2,
                              H3, guard, verdict band, diagnostics incl. answer-only correctness (grader verdict
                              alone, no citation rule) per system and era, paraphrase stability
  chunk-compare --questions P dev-only diagnostic: recall@5 of chunk=unit vs fixed 500-token chunks (50-token
                              overlap, cut from the concatenated unit texts of each document), for BOTH the BM25
                              and the dense retriever (re-embedding costs well under 0.01 EUR). A fixed chunk
                              covers every unit whose text it overlaps; a gold ID counts as retrieved if any of
                              the 5 retrieved chunks covers it (this favours fixed chunks slightly; say so)
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError
