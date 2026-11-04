"""Partial runs, one budget per eval, source checks before ingest, strict verdicts, the grader's stripped
input, index staleness, error answers without messages, and the shared answer rules."""

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from rag_ai_act import cli, config, ingest
from rag_ai_act.budget import EUR_PER_USD, PRICES_USD_PER_M, RESERVE_SAFETY, Budget, BudgetExceeded
from rag_ai_act.grading import STRIPPED_KEYS, export_blind, export_stripped, load_verdicts
from rag_ai_act.index import DenseRetriever
from rag_ai_act.ingest import CORPUS, SourceCheckError, build_corpus, verify_sources
from rag_ai_act.metrics import missing_answers, summarize
from rag_ai_act.systems import ANSWER_RULES, agent_system_prompt, answer_prompt, run_agent, run_rag, run_s0
from rag_ai_act.tools import DateClauses, ProvisionStore
from tests.fakes import ScriptedChat, WordEmbeddings, chat, response

ROOT = Path(__file__).resolve().parents[1]
FINAL = json.dumps({"answer": "Yes, from the sources.", "cited_ids": ["AIA:Art6(2)"], "declined": False})


def item(qid, kind="single", era="2026", ids=("AIA:Art6(2)",)):
    return {"qid": qid, "question": f"Question {qid}?", "type": kind, "era": era, "gold_answer": "Expected",
            "gold_ids": list(ids) if kind != "unanswerable" else []}


def record(system, qid, cited=("AIA:Art6(2)",), declined=False, retrieved=()):
    return {"qid": qid, "system": system, "answer": "x", "cited_ids": list(cited), "declined": declined,
            "valid": True, "retrieved_ids": list(retrieved), "trace": [], "usage": {"eur": 0.01}, "latency_s": 0.1}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------------------------------ 1. partial runs


def test_missing_answers_count_as_failures_with_full_denominators():
    items = [item("q1"), item("q2", era="2024"), item("u1", "unanswerable", "n/a")]
    answers = {"S1": [record("S1", "q1", retrieved=["AIA:Art6(2)"])]}
    s = summarize(items, answers, {("S1", "q1"): True})["S1"]
    assert s["H1"] == {"n": 1, "of": 2, "pct": 50.0}
    assert s["h1_by_era"]["2024"] == {"n": 0, "of": 1, "pct": 0.0}
    assert s["H3"] == {"n": 0, "of": 1, "pct": 0.0}
    assert s["false_declines"] == {"n": 1, "of": 2, "pct": 50.0}
    assert s["recall_at_5"] == pytest.approx(0.5)
    assert s["missing"] == 2 and s["invalid"] == 0
    draft = summarize(items, answers, None)["S1"]
    assert draft["H3"]["of"] == 1 and draft["false_declines"]["of"] == 2 and draft["missing"] == 2
    assert missing_answers(items, answers) == [("S1", "q2"), ("S1", "u1")]


@pytest.fixture
def partial_results(tmp_path):
    items = [item("Q1"), item("Q2"), item("U1", "unanswerable", "n/a")]
    write_jsonl(tmp_path / "items.jsonl", items)
    answers = {}
    for system in ("S0", "S1", "S2", "S3"):
        qids = ["Q1"] if system == "S3" else ["Q1", "Q2", "U1"]  # S3 stopped at the budget after Q1
        answers[system] = [record(system, q, declined=(q == "U1")) for q in qids]
        write_jsonl(tmp_path / f"answers_{system}.jsonl", answers[system])
    export_blind(items, answers, tmp_path, seed=3)
    key = json.loads((tmp_path / "blind_key.json").read_text())["map"]
    write_jsonl(tmp_path / "verdicts_1.jsonl", [{"blind_id": b, "correct": True, "reason": ""} for b in key])
    meta = {"systems": ["S0", "S1", "S2", "S3"], "budget": {"spent_eur": 2.9, "warned": True, "stopped": True}}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_report_on_a_partial_run_has_no_band_and_counts_missing_as_wrong(partial_results):
    assert cli.main(["report", "--results", str(partial_results)]) == 0
    rep = json.loads((partial_results / "report.json").read_text())
    assert rep["band"] == "incomplete" and rep["missing_answers"] == 2
    s3 = rep["summary"]["S3"]
    assert (s3["H1"]["n"], s3["H1"]["of"]) == (1, 2)
    assert (s3["H3"]["n"], s3["H3"]["of"]) == (0, 1)
    assert (rep["answer_only"]["S3"]["all"]["n"], rep["answer_only"]["S3"]["all"]["of"]) == (1, 2)
    text = (partial_results / "report.md").read_text()
    assert "INCOMPLETE" in text and "incomplete" in text.split("## Verdict band", 1)[1]


def test_report_band_is_incomplete_even_without_a_stop_flag(partial_results):
    (partial_results / "meta.json").unlink()
    assert cli.main(["report", "--results", str(partial_results)]) == 0
    assert json.loads((partial_results / "report.json").read_text())["band"] == "incomplete"


# ------------------------------------------------------------------ eval harness (offline fakes)


@pytest.fixture
def eval_env(tmp_path, monkeypatch, mini_units, mini_chunks):
    processed = tmp_path / "processed"
    processed.mkdir()
    write_jsonl(processed / "units.jsonl", [asdict(u) for u in mini_units])
    write_jsonl(processed / "chunks.jsonl", [asdict(c) for c in mini_chunks])
    monkeypatch.setattr(cli, "PROCESSED_DIR", processed)
    monkeypatch.setattr(cli, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "make_llm", lambda *a, **k: chat([response(FINAL)]))
    monkeypatch.setattr(config, "make_embeddings", lambda: WordEmbeddings())
    monkeypatch.setattr(config, "run_metadata", dict)
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [item("q1"), item("u1", "unanswerable", "n/a")])
    return {"tmp": tmp_path, "questions": questions, "processed": processed}


def run_eval(env, out, systems):
    code = cli.main(["eval", "--questions", str(env["questions"]), "--systems", systems, "--out", str(out)])
    return code, json.loads((out / "meta.json").read_text())


def test_eval_builds_a_missing_index_inside_its_budget_and_splits_the_spend(eval_env):
    code, meta = run_eval(eval_env, eval_env["tmp"] / "run1", "S2")
    assert code == 0
    b = meta["budget"]
    assert b["index_eur"] > 0 and b["answers_eur"] > 0
    assert b["spent_eur"] == pytest.approx(b["index_eur"] + b["answers_eur"])
    assert (eval_env["tmp"] / "index" / "index.faiss").is_file()
    code, meta = run_eval(eval_env, eval_env["tmp"] / "run2", "S2")  # the saved index matches: no rebuild
    assert code == 0 and meta["budget"]["index_eur"] == 0 and meta["budget"]["answers_eur"] > 0


def test_eval_rebuilds_a_stale_index_inside_the_same_budget(eval_env, mini_chunks, capsys):
    run_eval(eval_env, eval_env["tmp"] / "run1", "S2")
    changed = [replace(c, text=c.text + " amended") if n == 0 else c for n, c in enumerate(mini_chunks)]
    write_jsonl(eval_env["processed"] / "chunks.jsonl", [asdict(c) for c in changed])
    code, meta = run_eval(eval_env, eval_env["tmp"] / "run2", "S2")
    assert code == 0 and meta["budget"]["index_eur"] > 0
    assert "saved index not used" in capsys.readouterr().err


def test_eval_stopped_at_the_budget_writes_an_incomplete_draft(eval_env, monkeypatch):
    class StopAfterTwoChats(Budget):
        def reserve_chat(self, prompt_tokens, max_tokens):
            if sum(r["kind"] == "chat" for r in self.log) >= 2:
                raise BudgetExceeded("synthetic stop")
            return super().reserve_chat(prompt_tokens, max_tokens)

    monkeypatch.setattr(cli, "Budget", StopAfterTwoChats)
    out = eval_env["tmp"] / "run"
    code, meta = run_eval(eval_env, out, "S0,S1")
    assert code == 2 and meta["budget"]["stopped"] is True and meta["missing_answers"] == 2
    assert meta["budget"]["index_eur"] == 0
    draft = (out / "draft_report.md").read_text()
    assert "INCOMPLETE run: 2 of 4 answers are missing" in draft
    s1_row = next(line for line in draft.splitlines() if line.startswith("| S1 |"))
    assert "0/1 (0.0%)" in s1_row and "1/1 (100.0%)" in s1_row  # H3 failed, false decline counted
    assert "| 2 |" in s1_row  # missing column


# ------------------------------------------------------------------ 2. budget


def test_commit_past_the_stop_raises_and_keeps_the_cost():
    budget = Budget(warn_eur=10, stop_eur=0.0001)
    reservation = budget.reserve_chat(10, 0)
    with pytest.raises(BudgetExceeded):
        budget.commit(reservation, "chat", tokens_in=1_000_000, tokens_out=0)
    assert budget.spent_eur == pytest.approx(PRICES_USD_PER_M["chat_in"] * EUR_PER_USD)
    assert len(budget.log) == 1 and budget.reserved_eur == 0


def test_reservations_carry_the_safety_factor():
    cost = PRICES_USD_PER_M["embed_in"] * EUR_PER_USD  # one million embedding tokens
    assert RESERVE_SAFETY == 1.25
    with pytest.raises(BudgetExceeded):
        Budget(warn_eur=10, stop_eur=cost * 1.2).reserve_embed(1_000_000)
    assert Budget(warn_eur=10, stop_eur=cost * 1.3).reserve_embed(1_000_000) == pytest.approx(cost)


def test_make_eval_does_not_run_index_as_a_separate_process():
    out = subprocess.run(["make", "-n", "eval", "HELDOUT=q.jsonl"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    assert "rag_ai_act eval" in out and "rag_ai_act index" not in out


# ------------------------------------------------------------------ 3. source checks before ingest


@pytest.fixture
def raw_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    hashes = {}
    for n, name in enumerate(CORPUS):
        (raw / name).write_bytes(f"file {n}".encode())
        hashes[name] = hashlib.sha256(f"file {n}".encode()).hexdigest()
    return raw, hashes


def test_verify_sources_reports_mismatch_and_missing(raw_corpus):
    raw, hashes = raw_corpus
    verify_sources(raw, hashes)
    first, second = list(CORPUS)[:2]
    (raw / first).write_bytes(b"changed")
    (raw / second).unlink()
    with pytest.raises(SourceCheckError) as exc:
        verify_sources(raw, hashes)
    assert exc.value.problems == [f"{first}: sha256 differs", f"{second}: file missing"]


def test_build_corpus_checks_hashes_before_extraction(raw_corpus, monkeypatch, tmp_path):
    raw, hashes = raw_corpus
    monkeypatch.setattr(ingest, "extract_text", lambda path: pytest.fail("extracted before the source check"))
    with pytest.raises(SourceCheckError):
        build_corpus(raw, tmp_path / "out", expected_sha256={**hashes, next(iter(CORPUS)): "0" * 64})
    assert not (tmp_path / "out").exists()


def test_cli_ingest_stops_on_a_changed_file(raw_corpus, monkeypatch, tmp_path, capsys):
    raw, hashes = raw_corpus
    rows = "".join(f"| `{name}` | role | https://example.invalid/{n} | x | `{digest}` | lic |\n"
                   for n, (name, digest) in enumerate(hashes.items()))
    sources = tmp_path / "SOURCES.md"
    sources.write_text("| File | Role | URL | Retrieved | SHA-256 | Licence |\n|---|---|---|---|---|---|\n" + rows)
    (raw / list(CORPUS)[2]).write_bytes(b"tampered")
    monkeypatch.setattr(cli, "RAW_DIR", raw)
    monkeypatch.setattr(cli, "SOURCES_MD", sources)
    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(ingest, "extract_text", lambda path: pytest.fail("extracted before the source check"))
    assert cli.main(["ingest"]) == 1
    err = capsys.readouterr().err
    assert f"{list(CORPUS)[2]}: sha256 differs" in err and "nothing extracted" in err
    assert not (tmp_path / "processed").exists()


# ------------------------------------------------------------------ 4. strict verdicts


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_verdict_correct_must_be_a_json_boolean(tmp_path, value):
    key = tmp_path / "blind_key.json"
    key.write_text(json.dumps({"seed": 1, "map": {"B001": ["S0", "q1"], "B002": ["S1", "q1"]}}))
    path = tmp_path / "verdicts_1.jsonl"
    write_jsonl(path, [{"blind_id": "B001", "correct": True}, {"blind_id": "B002", "correct": value}])
    with pytest.raises(ValueError, match="B002"):
        load_verdicts(path, key)


def test_verdict_without_correct_is_rejected(tmp_path):
    key = tmp_path / "blind_key.json"
    key.write_text(json.dumps({"seed": 1, "map": {"B001": ["S0", "q1"]}}))
    path = tmp_path / "verdicts_1.jsonl"
    write_jsonl(path, [{"blind_id": "B001", "reason": "no field"}])
    with pytest.raises(ValueError, match="B001"):
        load_verdicts(path, key)


# ------------------------------------------------------------------ 5. stripped grading input


@pytest.fixture
def blind_results(tmp_path):
    results = tmp_path / "results"
    items = [item("q1"), item("q2", "multi", ids=("AIA:Art6(2)", "GLHR:para204"))]
    answers = {s: [record(s, "q1"), record(s, "q2")] for s in ("S0", "S3")}
    export_blind(items, answers, results, seed=5)
    return results


def test_grading_export_writes_only_the_grader_fields(blind_results, tmp_path):
    out = tmp_path / "grading"
    assert cli.main(["grading-export", "--results", str(blind_results), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["blind_stripped.jsonl"]
    rows = [json.loads(line) for line in (out / "blind_stripped.jsonl").read_text().splitlines()]
    blind = [json.loads(line) for line in (blind_results / "blind.jsonl").read_text().splitlines()]
    assert [tuple(r) for r in rows] == [STRIPPED_KEYS] * len(blind)
    assert [r["blind_id"] for r in rows] == [b["blind_id"] for b in blind]
    assert "gold_ids" not in rows[0] and "cited_ids" not in rows[0]


@pytest.mark.parametrize("where", ["non_empty", "results_dir", "inside_results", "parent_of_results"])
def test_grading_export_refuses_unsafe_folders(blind_results, tmp_path, where):
    out = {
        "non_empty": tmp_path / "grading",
        "results_dir": blind_results,
        "inside_results": blind_results / "grading",
        "parent_of_results": tmp_path,
    }[where]
    if where == "non_empty":
        out.mkdir()
        (out / "notes.txt").write_text("x")
    with pytest.raises(ValueError):
        export_stripped(blind_results / "blind.jsonl", out)
    assert cli.main(["grading-export", "--results", str(blind_results), "--out", str(out)]) == 1
    assert not (out / "blind_stripped.jsonl").exists()


def test_cli_help_lists_grading_export():
    proc = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "rag_ai_act", "--help"], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    assert "grading-export" in proc.stdout


# ------------------------------------------------------------------ 6. index staleness


class NamedEmbeddings(WordEmbeddings):
    def __init__(self, deployment):
        self.deployment = deployment


def test_saved_index_checks_chunk_texts_and_deployment(tmp_path, mini_chunks):
    path = tmp_path / "index"
    DenseRetriever(mini_chunks, config.BudgetedEmbeddings(NamedEmbeddings("embed-a"), Budget())).save(path)
    meta = json.loads((path / "chunks.json").read_text())
    assert meta["embed_deployment"] == "embed-a" and len(meta["chunks_sha256"]) == 64
    DenseRetriever.load(path, mini_chunks, NamedEmbeddings("embed-a"))
    changed = [replace(c, text=c.text + " amended") if n == 3 else c for n, c in enumerate(mini_chunks)]
    with pytest.raises(ValueError, match="chunk texts"):
        DenseRetriever.load(path, changed, NamedEmbeddings("embed-a"))
    with pytest.raises(ValueError, match="deployment"):
        DenseRetriever.load(path, mini_chunks, NamedEmbeddings("embed-b"))
    del meta["chunks_sha256"]
    (path / "chunks.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        DenseRetriever.load(path, mini_chunks, NamedEmbeddings("embed-a"))


# ------------------------------------------------------------------ 7. error answers without messages


SECRET = "request to https://service.invalid/deployments?sig=dummy-secret failed"


def broken(*args, **kwargs):
    raise RuntimeError(SECRET)


@pytest.mark.parametrize("system", ["S0", "S1", "S3"])
def test_error_answers_record_the_class_name_only(system, mini_units, mini_chunks, monkeypatch):
    from rag_ai_act.index import BM25Retriever

    model = chat([response(FINAL)])
    monkeypatch.setattr(ScriptedChat, "_generate", broken)
    if system == "S0":
        answer = run_s0("Question?", model, Budget())
    elif system == "S1":
        answer = run_rag("Question?", model, BM25Retriever(mini_chunks), Budget())
    else:
        answer = run_agent("Question?", model, BM25Retriever(mini_chunks), ProvisionStore(mini_units),
                           DateClauses(mini_units), Budget())
    assert answer.valid is False and answer.answer == "error: RuntimeError"
    assert "dummy-secret" not in json.dumps(asdict(answer))


def test_failed_query_embedding_records_the_class_name_only(mini_chunks):
    retriever = DenseRetriever(mini_chunks, WordEmbeddings())

    class FailingQueries(WordEmbeddings):
        def embed_query(self, text):
            raise RuntimeError(SECRET)

    retriever.embeddings = FailingQueries()
    answer = run_rag("Question?", chat([response(FINAL)]), retriever, Budget())
    assert answer.valid is False and answer.answer == "error: RuntimeError"


# ------------------------------------------------------------------ answer rules


def test_answer_rules_multi_part_citation_and_decline_rule():
    assert ("Cite every source your answer relies on and no other. If the question has several parts, cite the "
            "source for each part.") in ANSWER_RULES
    assert ('4. If the sources do not state the specific fact the question asks for, set "declined" to true, '
            'leave "cited_ids" empty, and say in "answer" that the sources do not answer the question; you may '
            'add what the sources do state. Do not guess.') in ANSWER_RULES
    assert "do not contain the answer" not in ANSWER_RULES
    for prompt in (agent_system_prompt(), answer_prompt("Q?", None, True), answer_prompt("Q?", "ctx", False)):
        assert prompt.startswith(ANSWER_RULES)
