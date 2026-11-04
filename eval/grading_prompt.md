# Grading prompt (blind correctness grading)

Used verbatim for every grading session. The grader runs in an empty folder that contains only `blind_stripped.jsonl`: one JSON object per line with `blind_id`, `question`, `gold_answer`, `answer`, `declined`. It never sees which system produced an answer, the question IDs, cited IDs or the grading key. Citations are scored separately and mechanically.

---

You are grading answers to questions about the EU AI Act and the European Commission's guidelines on it. Read `blind_stripped.jsonl` in the current folder. For every line, decide whether `answer` is correct when compared with `gold_answer`.

Rules:
1. Judge only against `gold_answer`. Do not use your own knowledge of the law to accept or reject an answer; if the gold answer says X, an answer saying not-X is wrong even if you believe not-X.
2. An answer is correct if it gives the same core conclusion as the gold answer (for example the same yes/no, the same date, the same provision or obligation) and adds nothing that contradicts it. Extra correct detail, different wording and hedges such as "according to the draft guidelines" are fine.
3. An answer is wrong if its core conclusion differs, if it is missing the part the question asks for (for example the date in a "from when" question), or if it contains a statement that contradicts the gold answer.
4. If `declined` is true, the answer is wrong (these are all answerable questions).
5. Do not reward length or confidence.

Output: write nothing to disk. Your final message is exactly one JSON object per line, one line per input line, in input order, and nothing else:
{"blind_id": "<id>", "correct": true or false, "reason": "<one short sentence>"}
