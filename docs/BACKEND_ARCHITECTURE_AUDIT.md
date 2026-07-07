# MeetScribe Backend Architecture Audit

## Scope

This audit documents the current backend architecture after migration from the
legacy Agentic/Gemini Minutes of Meeting path to the ML-based MoM pipeline. It
is intentionally documentation-first: no files are deleted, and legacy files are
classified for later manual review.

## A. Current Execution Flow

### Audio Upload

1. `app/main.py`
2. `transcription/audio_utils.py`
   - Validates and preprocesses uploaded audio.
3. `transcription/sarvam_client.py`
   - Sends audio to Sarvam and receives transcription segments.
4. `transcription/speaker_resolution.py`
   - Detects and prepares speaker labels.
5. `app/main.py`
   - Renders meeting information, speaker review, and transcript review.
6. `transcription/transcript_editing.py`
   - Applies reviewed transcript edits.
7. `app/main.py::run_meeting_analysis`
   - Runs the production ML MoM pipeline.
8. `summarization/base_summarizer.py`
   - Provides the stable `MeetingAnalysisResult` contract.
9. `exports/pdf_exporter.py`, `exports/docx_exporter.py`,
   `exports/email_sender.py`
   - Reuse the same output contract for PDF, DOCX, and email.

### Transcript Upload

1. `app/main.py`
2. `transcription/transcript_file_utils.py`
   - Extracts text from TXT, PDF, or DOCX input.
3. `transcription/speaker_mapping.py` and
   `transcription/speaker_resolution.py`
   - Preserve speaker review behavior.
4. `app/main.py`
   - Runs the same meeting information, speaker review, transcript review, and
   ML MoM generation path as audio uploads.

## B. Production ML Execution Flow

`Reviewed Transcript`
-> `ml_mom/transcript_parser.py`
-> `ml_mom/preprocessing.py`
-> `ml_mom/feature_extraction.py`
-> `ml_mom/embeddings.py`
-> `ml_mom/clustering.py`
-> `ml_mom/predict_ann.py`
-> `ml_mom/ann_model.py`
-> `ml_mom/mom_generator.py`
-> `summarization.base_summarizer.MeetingAnalysisResult`
-> `app/main.py`
-> `exports/pdf_exporter.py`
-> `exports/docx_exporter.py`
-> `exports/email_sender.py`

## C. ML Pipeline Files

- `ml_mom/transcript_parser.py`
  Runtime parser for speaker-labeled transcripts. Fully implemented and called
  from production report generation.
- `ml_mom/preprocessing.py`
  Runtime conservative text normalization. Fully implemented and called from
  production report generation.
- `ml_mom/feature_extraction.py`
  Runtime sentence feature extraction. Fully implemented and called from
  production report generation.
- `ml_mom/embeddings.py`
  Runtime local embedding generation using `sentence-transformers`. Fully
  implemented and called from production report generation.
- `ml_mom/clustering.py`
  Runtime topic grouping over sentence embeddings. Fully implemented and called
  from production report generation.
- `ml_mom/ann_model.py`
  Runtime ANN architecture loaded by prediction utilities.
- `ml_mom/predict_ann.py`
  Runtime prediction helpers plus standalone CLI utility.
- `ml_mom/mom_generator.py`
  Runtime rule-based MoM generator.

## D. Agentic Files

- `agentic/workflow.py`
  Legacy orchestration pipeline for MoM generation.
- `agentic/nodes.py`
  Legacy nodes: metadata, summary, discussion, decisions, action items, and
  validation.
- `agentic/state.py`
  Legacy state schema for Agentic workflow execution.
- `agentic/__init__.py`
  Legacy package export.

Current status: restored and retained for review, but production
`app/main.py` does not import or call them.

## E. Gemini-Only Files

- `llm_clients/gemini_client.py`
  Gemini API wrapper used by the old LLM summarizer.
- `summarization/llm_summarizer.py`
  Legacy prompt-driven LLM summarization implementation.
- `prompts/cleanup/`
  Legacy transcript cleanup prompt.
- `prompts/summarization/`
  Legacy meeting-summary and full-analysis prompts.
- `prompts/extraction/`
  Legacy action, decision, and discussion extraction prompts.
- `temp_test_gemini_integration.py`
  Legacy Gemini smoke test.
- `temp_test_full_pipeline.py`
  Legacy Gemini full-pipeline test.

Current status: not used by production MoM generation.

## F. Runtime-Only Files

- `app/main.py`
- `config/settings.py`
- `transcription/audio_utils.py`
- `transcription/sarvam_client.py`
- `transcription/speaker_mapping.py`
- `transcription/speaker_resolution.py`
- `transcription/transcript_editing.py`
- `transcription/transcript_file_utils.py`
- `exports/pdf_exporter.py`
- `exports/docx_exporter.py`
- `exports/email_sender.py`
- Runtime ML modules listed in section C.
- `summarization/base_summarizer.py`

## G. Training-Only and Dataset Files

- `ml_mom/annotation_tool.py`
- `ml_mom/annotation_app.py`
- `ml_mom/merge_annotations.py`
- `ml_mom/training_dataset.py`
- `ml_mom/train_ann.py`
- `datasets/annotations/*.csv`
- `datasets/processed/master_dataset.csv`
- `datasets/raw_transcripts/*.txt`

These files support supervised dataset creation and ANN training. They should
not run during normal Streamlit report generation.

## H. Duplicate Models Found

- `summarization.base_summarizer.MeetingAnalysisResult`
  Production contract used by `app/main.py`, PDF export, DOCX export, and email.
- `meeting_analysis.schemas.MeetingAnalysisResult`
  Skeleton architecture schema retained from the earlier design phase.

Recommendation: keep both for now. The production contract is the
`summarization.base_summarizer` version. The `meeting_analysis` schema can be
reviewed later if the project adopts `meeting_analysis/` as a formal service
boundary.

## I. Files That Appear Unused in Production

- `app/main_old.py`
  Legacy Streamlit entry-point snapshot with Gemini-based report generation.
- `agentic/`
  Legacy MoM orchestration.
- `llm_clients/`
  Legacy Gemini client package.
- `summarization/llm_summarizer.py`
  Legacy LLM summarizer.
- `prompts/cleanup/`, `prompts/summarization/`, `prompts/extraction/`
  Legacy LLM prompt templates.
- `meeting_analysis/service.py`, `meeting_analysis/schemas.py`,
  `meeting_analysis/validation.py`
  Skeleton service boundary, not currently used by production `app/main.py`.
- `ml_mom/pipeline.py`, `ml_mom/classification.py`,
  `ml_mom/template_generator.py`
  Placeholder modules from the skeleton phase; the implemented runtime uses
  `predict_ann.py` and `mom_generator.py` instead.
- `*_backup/`
  Backup copies of legacy directories. Retained because no files should be
  deleted automatically.
- `temp_test_*.py`
  Local smoke/demo tests from earlier development phases.

## J. Safe-To-Delete-Later Recommendations

Do not delete automatically. After manual verification, these can be considered:

- `agentic/`: safe to delete after confirming no rollback need.
- `llm_clients/`: safe to delete if Gemini is permanently retired.
- `summarization/llm_summarizer.py`: safe to delete after legacy tests are
  removed or updated.
- `prompts/cleanup/`, `prompts/summarization/`, `prompts/extraction/`: safe to
  delete with the LLM summarizer.
- `app/main_old.py`: safe to delete if no rollback snapshot is required.
- `agentic_backup/`, `llm_clients_backup/`, `summarization_backup/`: safe to
  delete manually after verifying the original tracked folders are present.
- `temp_test_gemini_integration.py`, `temp_test_full_pipeline.py`, and other
  Gemini/LLM temp tests: safe to delete or archive after migration sign-off.

## K. Production Dependency Check

Production paths checked:

- `app/main.py`
- `exports/`
- `transcription/`
- `ml_mom/`
- `meeting_analysis/`
- `email_agent/`
- `config/`

Result:

- No production import of `agentic`.
- No production import of `llm_clients`.
- No production import of `summarization.llm_summarizer`.
- No production call to `GeminiClient`.
- No production call to `LLMSummarizer`.
- No production call to `run_meeting_analysis_workflow`.

Legacy references remain only in legacy/backward-compatibility files and local
test scripts.
