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

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from rag_ai_act.budget import Budget, BudgetExceeded
from rag_ai_act.grading import agreement, export_blind, load_verdicts
from rag_ai_act.ids import matches
from rag_ai_act.index import TOP_K, BM25Retriever, DenseRetriever
from rag_ai_act.ingest import Chunk, Unit, build_corpus
from rag_ai_act.metrics import paraphrase_stability, strict_correct, summarize, verdict_band

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
SOURCES_MD = ROOT / "data" / "SOURCES.md"
SYSTEM_NAMES = ("S0", "S1", "S2", "S3")
FIXED_CHUNK_TOKENS = 500
FIXED_CHUNK_OVERLAP = 50


# ------------------------------------------------------------------------------------------------ helpers


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_corpus() -> tuple[list[Unit], list[Chunk]]:
    units_path, chunks_path = PROCESSED_DIR / "units.jsonl", PROCESSED_DIR / "chunks.jsonl"
    if not units_path.is_file() or not chunks_path.is_file():
        raise SystemExit("data/processed is missing: run `python -m rag_ai_act ingest` first")
    return [Unit(**r) for r in _read_jsonl(units_path)], [Chunk(**r) for r in _read_jsonl(chunks_path)]


def _dense(chunks: list[Chunk], embeddings, save: bool) -> DenseRetriever:
    """The saved index when it matches the chunks (no document embeddings), else a fresh build."""
    if (INDEX_DIR / "index.faiss").is_file():
        try:
            return DenseRetriever.load(INDEX_DIR, chunks, embeddings)
        except ValueError as exc:
            print(f"saved index not used: {exc}", file=sys.stderr)
    retriever = DenseRetriever(chunks, embeddings)
    if save:
        retriever.save(INDEX_DIR)
    return retriever


def _count(c: dict | None) -> str:
    return "-" if c is None else f"{c['n']}/{c['of']} ({c['pct']}%)"


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _summary_table(summary: dict) -> list[str]:
    lines = [
        "| system | H1 | H2 | H3 | false declines | recall@5 | evidence recall | citation precision | multi-hop | invalid | EUR |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for system, s in summary.items():
        cp = s["citation_precision"]
        lines.append(
            f"| {system} | {_count(s['H1'])} | {_count(s['H2'])} | {_count(s['H3'])} | {_count(s['false_declines'])} "
            f"| {_num(s['recall_at_5'])} | {_num(s['evidence_recall'])} "
            f"| {_num(cp['mean'])} (citing nothing: {cp['n_citing_nothing']}) | {_count(s['multi_hop'])} "
            f"| {s['invalid']} | {s['eur']:.4f} |"
        )
    return lines


# ------------------------------------------------------------------------------------------------ fetch


def _parse_sources(text: str) -> list[dict]:
    """Rows of the table in SOURCES.md: file name (in backticks), URL, sha256."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        name = next((c.strip("`") for c in cells if c.startswith("`") and c.lower().rstrip("`").endswith(".pdf")), None)
        digest = next((c.strip("`") for c in cells if len(c.strip("`")) == 64 and c.strip("`").isalnum()), None)
        url = next((c.strip("<>") for c in cells if c.startswith(("http://", "https://", "<http"))), None)
        if name and digest:
            rows.append({"name": name, "sha256": digest.lower(), "url": url})
    return rows


def cmd_fetch(args) -> int:
    rows = _parse_sources(SOURCES_MD.read_text(encoding="utf-8"))
    if not rows:
        print("no sources found in data/SOURCES.md", file=sys.stderr)
        return 1
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for row in rows:
        path = RAW_DIR / row["name"]
        scripted = bool(row["url"]) and "eur-lex" not in row["url"]
        if not path.is_file() and scripted:
            try:
                request = urllib.request.Request(row["url"], headers={"User-Agent": "rag-ai-act-fetch"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    path.write_bytes(response.read())
                print(f"downloaded {row['name']}")
            except OSError as exc:
                print(f"MISSING  {row['name']}: download failed ({exc})")
                continue
        if not path.is_file():
            print(f"MISSING  {row['name']}: download manually from {row['url']} and save it as data/raw/{row['name']}")
        elif _sha256(path) == row["sha256"]:
            print(f"OK       {row['name']}")
            ok += 1
        else:
            print(f"MISMATCH {row['name']}: sha256 differs from data/SOURCES.md")
    print(f"{ok}/{len(rows)} files OK")
    return 0 if ok == len(rows) else 1


# ------------------------------------------------------------------------------------------------ ingest, index


def cmd_ingest(args) -> int:
    units, chunks = build_corpus(RAW_DIR, PROCESSED_DIR)
    print(f"ingest: {len(units)} units, {len(chunks)} chunks written to data/processed (coverage check passed)")
    return 0


def cmd_index(args) -> int:
    from rag_ai_act.config import BudgetedEmbeddings, make_embeddings

    _, chunks = _load_corpus()
    budget = Budget()
    try:
        retriever = DenseRetriever(chunks, BudgetedEmbeddings(make_embeddings(), budget))
    except BudgetExceeded as exc:
        print(f"stopped at the budget: {exc}", file=sys.stderr)
        return 2
    retriever.save(INDEX_DIR)
    print(f"index: {len(chunks)} chunks embedded, saved to data/index, {budget.spent_eur:.4f} EUR")
    return 0


# ------------------------------------------------------------------------------------------------ eval


def _load_items(questions: Path, paraphrases: Path | None) -> list[dict]:
    items = _read_jsonl(questions)
    qids = {it["qid"] for it in items}
    if len(qids) != len(items):
        raise SystemExit("duplicate qid in the questions file")
    if paraphrases is not None:
        for rec in _read_jsonl(paraphrases):
            if rec["qid"] in qids:
                raise SystemExit(f"paraphrase qid {rec['qid']} collides with a question qid")
            if rec.get("paraphrase_of") not in qids:
                raise SystemExit(f"paraphrase {rec['qid']} points to an unknown question")
            items.append(rec)
    return items


def cmd_eval(args) -> int:
    from rag_ai_act.config import BudgetedEmbeddings, make_embeddings, make_llm, run_metadata
    from rag_ai_act.runner import run_questions
    from rag_ai_act.systems import run_agent, run_rag, run_s0
    from rag_ai_act.tools import DateClauses, ProvisionStore

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not names or any(n not in SYSTEM_NAMES for n in names):
        raise SystemExit(f"--systems must be a comma-separated subset of {','.join(SYSTEM_NAMES)}")
    questions = Path(args.questions)
    paraphrases = Path(args.paraphrases) if args.paraphrases else None
    items = _load_items(questions, paraphrases)
    out = Path(args.out) if args.out else ROOT / "results" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    units, chunks = _load_corpus()
    budget = Budget()
    start = time.perf_counter()
    llm = make_llm()
    meta = run_metadata()
    result = {"answers": {n: [] for n in names}, "stopped": False, "stopped_at": None}
    try:
        systems = {}
        needs_dense = "S2" in names or "S3" in names
        dense = _dense(chunks, BudgetedEmbeddings(make_embeddings(), budget), save=True) if needs_dense else None
        bm25 = BM25Retriever(chunks)
        store, dates = ProvisionStore(units), DateClauses(units)
        if "S0" in names:
            systems["S0"] = lambda q: run_s0(q, llm, budget)
        if "S1" in names:
            systems["S1"] = lambda q: run_rag(q, llm, bm25, budget)
        if "S2" in names:
            systems["S2"] = lambda q: run_rag(q, llm, dense, budget)
        if "S3" in names:
            systems["S3"] = lambda q: run_agent(q, llm, dense, store, dates, budget)
        result = run_questions(items, systems, budget)
    except BudgetExceeded:
        result["stopped"] = True
        result["stopped_at"] = {"system": "setup", "qid": None}

    answers = result["answers"]
    for name, records in answers.items():
        _write_jsonl(out / f"answers_{name}.jsonl", records)
    _write_jsonl(out / "items.jsonl", items)
    export_blind(items, answers, out)
    meta.update(
        {
            "budget": {"spent_eur": budget.spent_eur, "warned": budget.warned, "stopped": result["stopped"]},
            "stopped_at": result["stopped_at"],
            "systems": names,
            "questions_file": questions.name,
            "questions_sha256": _sha256(questions),
            "runtime_s": round(time.perf_counter() - start, 1),
        }
    )
    if paraphrases is not None:
        meta.update({"paraphrases_file": paraphrases.name, "paraphrases_sha256": _sha256(paraphrases)})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    summary = summarize(items, answers, None)
    lines = ["# Draft report (mechanical metrics only, no grading yet)", ""]
    if result["stopped"]:
        lines += [f"STOPPED at the budget: {result['stopped_at']}. Results are partial.", ""]
    lines += _summary_table(summary)
    (out / "draft_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"eval: {sum(len(r) for r in answers.values())} answers, {budget.spent_eur:.4f} EUR, written to {out}")
    if result["stopped"]:
        print(f"STOPPED at the budget at {result['stopped_at']}", file=sys.stderr)
        return 2
    return 0


# ------------------------------------------------------------------------------------------------ report


def _verdicts(results: Path, name: str) -> dict | None:
    path = results / f"{name}.jsonl"
    return load_verdicts(path, results / "blind_key.json") if path.is_file() else None


def build_report(results: Path) -> dict:
    """Everything the report shows, as a JSON-serialisable dict."""
    items = _read_jsonl(results / "items.jsonl")
    answers = {
        name: _read_jsonl(results / f"answers_{name}.jsonl")
        for name in SYSTEM_NAMES
        if (results / f"answers_{name}.jsonl").is_file()
    }
    v1, v2, final = (_verdicts(results, n) for n in ("verdicts_1", "verdicts_2", "verdicts_final"))
    correct = final if final is not None else v1
    if correct is None:
        raise SystemExit("no verdicts_final.jsonl or verdicts_1.jsonl in the results folder")
    summary = summarize(items, answers, correct)
    band = verdict_band(summary) if all(s in summary for s in ("S0", "S1", "S3")) else None
    main = [it for it in items if "paraphrase_of" not in it]
    answerable = [it for it in main if it["type"] != "unanswerable"]

    answer_only = {}
    for system, records in answers.items():
        by_qid = {r["qid"]: r for r in records}
        row = {}
        for label, subset in (
            ("all", answerable),
            ("2024", [i for i in answerable if i["era"] == "2024"]),
            ("2026", [i for i in answerable if i["era"] == "2026"]),
        ):
            have = [i for i in subset if i["qid"] in by_qid]
            n = sum(correct.get((system, i["qid"]), False) for i in have)
            row[label] = {"n": n, "of": len(have), "pct": round(100 * n / len(have), 1) if have else 0.0}
        answer_only[system] = row

    by_qid_item = {it["qid"]: it for it in items}
    stability = {}
    for system, records in answers.items():
        by_qid = {r["qid"]: r for r in records}
        orig, para = [], []
        for it in items:
            base = by_qid_item.get(it.get("paraphrase_of", ""))
            if base is None or it["qid"] not in by_qid or base["qid"] not in by_qid:
                continue
            para.append(strict_correct(it, by_qid[it["qid"]], correct.get((system, it["qid"]), False)))
            orig.append(strict_correct(base, by_qid[base["qid"]], correct.get((system, base["qid"]), False)))
        if para:
            stability[system] = {**paraphrase_stability(orig, para), "n_items": len(para)}

    agree = None
    if v1 is not None and v2 is not None:
        agree = agreement(v1, v2)
        agree["disagreements"] = [list(d) for d in agree["disagreements"]]
    meta_path = results / "meta.json"
    stopped = json.loads(meta_path.read_text(encoding="utf-8"))["budget"]["stopped"] if meta_path.is_file() else None
    return {
        "verdicts_source": "verdicts_final" if final is not None else "verdicts_1",
        "settled": final is not None,
        "summary": summary,
        "band": band,
        "agreement": agree,
        "answer_only": answer_only,
        "paraphrase_stability": stability,
        "run_stopped_at_budget": stopped,
    }


def render_report(rep: dict) -> str:
    lines = ["# Report", ""]
    if not rep["settled"]:
        lines += ["Grading status: unsettled (verdicts_1 only; no verdicts_final.jsonl).", ""]
    else:
        lines += ["Grading status: verdicts_final.jsonl (pass 1 with settled disagreements).", ""]
    if rep["run_stopped_at_budget"]:
        lines += ["The run stopped at the budget; results are partial.", ""]
    lines += ["## H1, H2, H3, guard", ""] + _summary_table(rep["summary"])
    lines += ["", "H1: strict accuracy on answerable questions (verdict correct, not declined, valid, right source IDs).",
              "H2: the same on 2026-era questions. H3: declined unanswerable questions.",
              "Guard: false declines among answerable questions.", ""]
    lines += ["## Verdict band", "", f"{rep['band'] or 'not computable (needs S0, S1 and S3)'}", ""]
    lines += ["## Grader agreement (pass 1 vs pass 2)", ""]
    a = rep["agreement"]
    lines += ["not available (needs verdicts_1 and verdicts_2)" if a is None
              else f"{_count(a)}; disagreements: {a['disagreements']}", ""]
    lines += ["## Diagnostics", "", "Answer-only correctness (grader verdict alone, no citation rule) next to strict H1:", "",
              "| system | answer-only all | 2024 | 2026 | strict H1 |", "|---|---|---|---|---|"]
    for system, row in rep["answer_only"].items():
        lines.append(f"| {system} | {_count(row['all'])} | {_count(row['2024'])} | {_count(row['2026'])} "
                     f"| {_count(rep['summary'][system]['H1'])} |")
    lines += ["", "Paraphrase stability (strict-correct, original vs paraphrase):", ""]
    if rep["paraphrase_stability"]:
        lines += ["| system | items | original | paraphrase | within 2 |", "|---|---|---|---|---|"]
        for system, p in rep["paraphrase_stability"].items():
            lines.append(f"| {system} | {p['n_items']} | {p['orig']} | {p['para']} | {p['within_2']} |")
    else:
        lines.append("no paraphrase items")
    return "\n".join(lines) + "\n"


def cmd_report(args) -> int:
    results = Path(args.results)
    rep = build_report(results)
    (results / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (results / "report.md").write_text(render_report(rep), encoding="utf-8")
    print(f"report: band {rep['band']}, written to {results}/report.md and report.json")
    return 0


# ------------------------------------------------------------------------------------------------ chunk-compare


def fixed_chunks(units: list[Unit]) -> tuple[list[Chunk], dict[str, list[str]]]:
    """Fixed-size chunks per document, cut from the concatenated unit texts. Returns the chunks and, for
    each chunk ID, the source IDs of every unit whose text the chunk overlaps."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    chunks: list[Chunk] = []
    covers: dict[str, list[str]] = {}
    for doc in sorted({u.doc for u in units}):
        doc_units = sorted((u for u in units if u.doc == doc), key=lambda u: u.order)
        tokens: list[int] = []
        owner: list[int] = []
        for i, unit in enumerate(doc_units):
            toks = enc.encode(unit.text + "\n")
            tokens.extend(toks)
            owner.extend([i] * len(toks))
        step = FIXED_CHUNK_TOKENS - FIXED_CHUNK_OVERLAP
        for n, begin in enumerate(range(0, max(len(tokens), 1), step)):
            window = slice(begin, begin + FIXED_CHUNK_TOKENS)
            ids = [doc_units[i].source_id for i in dict.fromkeys(owner[window])]
            if not ids:
                break
            chunk_id = f"{doc}:fixed{n}"
            chunks.append(Chunk(chunk_id, ids[0], doc, doc_units[owner[begin]].draft, 0, enc.decode(tokens[window])))
            covers[chunk_id] = ids
            if begin + FIXED_CHUNK_TOKENS >= len(tokens):
                break
    return chunks, covers


def cmd_chunk_compare(args) -> int:
    from rag_ai_act.config import BudgetedEmbeddings, make_embeddings

    items = [i for i in _read_jsonl(Path(args.questions)) if i["type"] != "unanswerable" and "paraphrase_of" not in i]
    units, chunks = _load_corpus()
    fixed, covers = fixed_chunks(units)
    budget = Budget()
    embeddings = BudgetedEmbeddings(make_embeddings(), budget)

    def recall(retriever, per_chunk) -> float:
        total = 0.0
        for it in items:
            covered = [sid for c in retriever.retrieve(it["question"], TOP_K) for sid in per_chunk(c)]
            total += sum(any(matches(c, g) for c in covered) for g in it["gold_ids"]) / len(it["gold_ids"])
        return total / len(items)

    try:
        retrievers = {
            "BM25": (BM25Retriever(chunks), BM25Retriever(fixed)),
            "dense": (_dense(chunks, embeddings, save=False), DenseRetriever(fixed, embeddings)),
        }
        rows = {
            name: (recall(unit_r, lambda c: [c.source_id]), recall(fixed_r, lambda c: covers[c.chunk_id]))
            for name, (unit_r, fixed_r) in retrievers.items()
        }
    except BudgetExceeded as exc:
        print(f"stopped at the budget: {exc}", file=sys.stderr)
        return 2
    print(f"chunk-compare on {len(items)} answerable questions: recall@{TOP_K}")
    print(f"unit chunks: {len(chunks)}; fixed chunks ({FIXED_CHUNK_TOKENS} tokens, {FIXED_CHUNK_OVERLAP} overlap): {len(fixed)}")
    print("| retriever | chunk = unit | fixed 500 |")
    print("|---|---|---|")
    for name, (a, b) in rows.items():
        print(f"| {name} | {a:.3f} | {b:.3f} |")
    print("A fixed chunk covers every unit its text overlaps and a gold ID counts if any of the 5 chunks covers it,")
    print("which favours fixed chunks slightly.")
    print(f"embedding cost: {budget.spent_eur:.4f} EUR")
    return 0


# ------------------------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_ai_act", description="Cited Q&A over the EU AI Act: build, run, evaluate.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="verify data/raw against data/SOURCES.md; download what can be scripted").set_defaults(func=cmd_fetch)
    sub.add_parser("ingest", help="data/raw -> data/processed (coverage-checked)").set_defaults(func=cmd_ingest)
    sub.add_parser("index", help="build the FAISS index in data/index/ (budget-capped)").set_defaults(func=cmd_index)
    p = sub.add_parser("eval", help="run the systems on a question set; write answers, blind export, meta, draft report")
    p.add_argument("--questions", required=True)
    p.add_argument("--paraphrases")
    p.add_argument("--systems", default=",".join(SYSTEM_NAMES))
    p.add_argument("--out")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("report", help="H1, H2, H3, guard, verdict band and diagnostics from graded results")
    p.add_argument("--results", required=True)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("chunk-compare", help="dev-only recall@5 of unit chunks vs fixed 500-token chunks")
    p.add_argument("--questions", required=True)
    p.set_defaults(func=cmd_chunk_compare)
    args = parser.parse_args(argv)
    return args.func(args)
