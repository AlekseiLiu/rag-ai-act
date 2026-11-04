# Test plan

The suite checks public behavior with small legal-text excerpts, toy evaluation records,
scripted chat messages and deterministic word-hash embeddings. Offline tests block socket
connections, DNS lookup and datagram sends. The cached `cl100k_base` encoding loads once
before the per-test network guard, which never applies to tests marked `live`. Scripted chat
models repeat their last message after the script ends. No credentials are required for the default run.
Network and runaway-model guards use `pytest.fail`, so normal exception recovery in a system
cannot convert a test guard failure into an invalid answer.

Run `.venv/bin/python -m pytest -q` and `.venv/bin/ruff check tests`.
Use `COLUMNS=400` when filtering short failure summaries so long test names do not truncate
the exception class. Unimplemented functions should fail openly; no test catches their
implementation placeholder exception or marks those failures as expected.

## Coverage matrix

Parameterized cases belong to the named test below.

| Row | File | Tests and behavior |
| --- | --- | --- |
| A | `tests/test_ids.py` | `test_canonical_round_trip`, `test_loose_and_subpoint_ids`, `test_parsed_fields_and_distinct_annex_sections`, `test_invalid_ids_raise`, `test_matching_is_directional_and_document_aware`, `test_invalid_gold_raises`, `test_parent_ids`: canonical and loose IDs, inserted numbers, subpoint reduction, annex section identity, invalid inputs, directional matching and parents. |
| B | `tests/test_ingest.py` | `test_clean_text_removes_markers_headers_and_joins_words`, `test_clean_text_removes_footnotes_and_page_footers`: cleaning with real extraction artifacts. |
| B | `tests/test_ingest.py` | `test_article_numbering_insertions_and_embedded_points`, `test_article3_definitions_are_individual_units`, `test_article113_unnumbered_paragraphs_keep_points_together`, `test_annex_point_and_subpoints`, `test_deleted_paragraph_does_not_create_a_unit`: article, definition, annex and deletion parsing. |
| B | `tests/test_ingest.py` | `test_annex_sections_keep_their_numbered_points`, `test_ordinary_examples_join_preceding_paragraph_inside_numbered_section`: Annexes I, VIII and XI produce section units containing all their points; ordinary examples remain in their numbered paragraphs, while only practical example blocks create section example units. |
| B | `tests/test_ingest.py` | `test_guidelines_page_break_and_all_example_boxes`, `test_stray_guidelines_number_is_not_a_paragraph`, `test_unit_identity_headers_and_draft_flags`: page continuations, one electricity example unit, number noise and metadata. |
| B | `tests/test_ingest.py` | `test_long_unit_splits_without_losing_text_or_identity`, `test_short_unit_is_one_chunk`, `test_coverage_checks_only_present_documents`, `test_coverage_names_missing_and_duplicate_ids`, `test_full_corpus_coverage`: chunk identity and size, content preservation independent of split-point whitespace, coverage diagnostics, present-document scope and full-corpus coverage. |
| C | `tests/test_tools.py` | `test_provision_exact_loose_and_article_order`, `test_unknown_provision_and_tool_not_found`: exact and loose lookups, article ordering, unknown IDs and actual tool output. |
| D | `tests/test_tools.py` | `test_dates_return_relevant_corpus_clause_only`, `test_unknown_date_query`, `test_three_tools_route_queries_to_real_data`, `test_source_has_no_date_literals`: dates from Articles 111/113, exclusion of other units, unknown query, tool routing and source scan. |
| E | `tests/test_answers.py` | `test_json_and_fenced_json`, `test_malformed_missing_and_wrong_typed_outputs_are_invalid`, `test_decline_clears_citations`, `test_loose_citations_normalized_and_unknown_retained`: parsing and answer validity. |
| F | `tests/test_metrics.py` | `test_evidence_single_alternative_vs_multi_all`, `test_strict_correct_requires_every_condition`, `test_recall_and_precision_are_directional_shares`: evidence requirements, correctness, recall and precision. |
| F | `tests/test_metrics.py` | `test_summary_denominators_era_declines_invalids_and_cost`, `test_summary_retrieval_uses_top_five_for_rag_and_all_for_agent`: explicit numerators, denominators, percentages, era split, unanswerable exclusion, invalids, precision population, retrieval diagnostics and None for diagnostics that do not apply to a system. |
| F | `tests/test_metrics.py` | `test_verdict_thresholds_and_useless_precedence`, `test_verdict_thresholds_scale_with_set_size`, `test_paraphrase_stability_counts_and_boundary`: all verdict bands, precedence, guard boundaries, scaled thresholds including exactly 1/20 improvement, and stability. |
| F | `tests/test_metrics.py` | `test_ungraded_summary_defers_correctness_metrics`, `test_summary_excludes_paraphrases_from_main_counts`: before grading, correctness metrics are None and mechanical metrics remain populated; paraphrase items do not affect the main counts. |
| G | `tests/test_systems_offline.py` | `test_s0_offline_answer_and_question_delivery`, `test_rag_offline_answer_retrieval_and_context`: real system entry points with a scripted chat model; S1/S2 use real indexes over excerpt-derived chunks and deliver five retrieved chunks to the model. |
| G | `tests/test_systems_offline.py` | `test_agent_executes_tools_records_returned_ids_and_parses_final`, `test_agent_bounds_actual_tool_calls_even_with_repeated_or_batched_requests`: all three tools execute, returned IDs match corpus units, tool messages reach the model, trace and deduplicated retrieval IDs are populated, actual calls and trace remain bounded. |
| G/H | `tests/test_systems_offline.py` | `test_s0_offline_answer_and_question_delivery`, `test_rag_offline_answer_retrieval_and_context`, `test_agent_executes_tools_records_returned_ids_and_parses_final`: each successful single-question run records positive input tokens and cost, reports the same cost as its named budget, and leaves no open reservations. `test_agent_bounds_actual_tool_calls_even_with_repeated_or_batched_requests` also limits model calls to the tool-call allowance plus two. |
| G/H | `tests/test_systems_offline.py` | `test_systems_propagate_budget_stop`, `test_agent_propagates_tool_budget_error`: every system propagates a reservation failure and a budget exception raised inside a fake model call; tool-raised budget exceptions also escape the agent. |
| G | `tests/test_systems_offline.py` | `test_agent_converts_service_errors_to_invalid_answers`, `test_baselines_convert_model_errors_to_invalid_answers`: ordinary model/tool failures produce invalid answers; tests require the failing fake to have actually executed. |
| H | `tests/test_budget_runner.py` | `test_reservations_stop_before_overspend`, `test_stop_includes_existing_spend_and_open_reservations`, `test_budget_warning_is_emitted_once`, `test_commit_releases_only_its_reservation_and_charges_actual_usage`: both call kinds, outstanding reservations, actual cost and warning behavior. |
| H | `tests/test_budget_runner.py` | `test_runner_preserves_completed_records_and_stops_every_system`, `test_metadata_excludes_key_and_has_reproducibility_fields`: partial results survive a mid-run stop; later systems never execute; metadata excludes a dummy key. |
| H | `tests/test_budget_runner.py` | `test_embedding_wrapper_accounts_for_each_input`, `test_embedding_wrapper_stops_before_service_call`, `test_embedding_wrapper_propagates_inner_budget_error`: document and query calls reserve before invoking the inner embeddings, commit actual usage as `embed` on each call, and propagate budget exceptions from either reservation or the wrapped call. |
| I | `tests/test_grading.py` | `test_blind_export_hides_labels_excludes_unanswerables_and_key_round_trips`, `test_blind_shuffle_is_reproducible_and_seed_sensitive`, `test_verdict_import_requires_exact_blind_ids`, `test_agreement_counts_disagreements_and_percentage`: hidden labels, reproducible shuffle, reverse mapping, rejected unknown/missing IDs and agreement. |
| I | `tests/test_grading.py` | `test_blind_export_includes_paraphrase_items`: paraphrased questions receive blind grading records alongside the original answerable questions. |
| I | `tests/test_grading.py` | `test_no_evaluation_answers_embedded_in_code`: when `HELDOUT_DIR` is set, read every JSONL file in that folder at test time, require at least one evaluation record and scan source/test files for whole-word IDs, questions and gold answers. No evaluation content is copied into the repository. |
| J | `tests/test_live_smoke.py` | `test_live_smoke_each_system`: four opt-in cases use the real client factories, budgeted embeddings, excerpt-derived chunks and the first development question; each system must return a valid answer. |

## Test selection

- Default selection excludes `live`. Run service smoke tests explicitly with
  `.venv/bin/python -m pytest -m live -k smoke`; they require configured service access.
- `test_full_corpus_coverage` has the `corpus` marker and skips when the required local
  corpus text is absent. All ordinary parsing and retrieval tests use checked-in excerpts.
- The optional evaluation-content scan skips when `HELDOUT_DIR` is unset. This is separate
  from the default corpus and service selections.
- The source date scan is executable without any implementation. Most behavioral tests
  initially fail at public function placeholders, including fixture setup for integration tests.

## Fixtures

Each text fixture is one contiguous, verbatim excerpt of a text extraction under `data/text/`;
whitespace, form feeds, footnotes and consolidation markers are retained. These are excerpts
of European Union legislation and European Commission guidelines. Legislation is reused
under the EUR-Lex reuse policy; Commission guideline excerpts are attributed to the European
Commission and reused under CC BY 4.0. Text has not been rewritten.

Line numbers below count newline-separated lines, including lines containing a form feed.

| Fixture | Source text | Lines |
| --- | --- | --- |
| `act_article3.txt` | `CELEX_02024R1689-20260727_EN_TXT.txt` | 246–365 |
| `act_article6.txt` | Same consolidated Act | 1071–1192 |
| `act_article10.txt` | Same consolidated Act | 1436–1511 |
| `act_article111.txt` | Same consolidated Act | 8320–8362 |
| `act_article113.txt` | Same consolidated Act | 8523–8560 |
| `act_annexiii.txt` | Same consolidated Act | 8744–8775 |
| `act_annexi_sections.txt` | Same consolidated Act | 8563–8697 |
| `act_annexviii_sections.txt` | Same consolidated Act | 9315–9415 |
| `act_annexxi_sections.txt` | Same consolidated Act | 9559–9631 |
| `guidelines_electricity.txt` | `Draft_Guidelines_on_the_classification_of_high_risk_AI_Annex_III_7MXR3YIz2GW3uPpJPWvvNDd8IOI_128561.txt` | 2384–2496 |
| `guidelines_number_noise.txt` | Same draft Annex III guidelines | 1964–2000 |
| `guidelines_inline_examples.txt` | Same draft Annex III guidelines | 384–640 |
| `definition_paragraph43.txt` | `Commission_Guidelines_on_the_definition_of_an_artificial_intelligence_system_established_by_Regulation_EU_20241689_AI_ActEnglish_nf2skCqfRtJdFgGJAvCODoPCWZ4_112455.txt` | 398–404 |

## Additional

- `test_definition_coverage_is_contiguous_to_maximum` (`tests/test_ingest.py`): checks gap detection for the definition guidelines independently of the other documents.
- `test_format_units_keeps_identity_text_and_draft_label` (`tests/test_tools.py`): prevents tool formatting from dropping draft status or source text.
- `test_prompt_includes_question_context_and_answer_contract` (`tests/test_answers.py`): ensures the shared prompt delivers the question, context, citation contract and decline/draft rules.
- `test_token_count_uses_cl100k` (`tests/test_budget_runner.py`): keeps cost estimation tied to the documented tokenizer.
- `test_runner_completes_without_a_budget_stop` (`tests/test_budget_runner.py`): distinguishes normal completion from partial results.
- `test_retrieval_returns_relevant_real_chunks_and_respects_k` (`tests/test_index.py`): prevents indexes from returning arbitrary fixture chunks or ignoring a requested result count.
- `test_dense_save_load_preserves_ranking` (`tests/test_index.py`): verifies index persistence through the public API.
- `test_rag_malformed_model_output_does_not_crash` (`tests/test_systems_offline.py`): checks invalid JSON handling at the system boundary, including preserved retrieval IDs.
