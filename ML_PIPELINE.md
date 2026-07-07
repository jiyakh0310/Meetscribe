# ML Pipeline

## Purpose

This document describes the production ML-based Minutes of Meeting pipeline used
by MeetScribe. The pipeline replaces the earlier Agentic/Gemini MoM generation
path while preserving the same `MeetingAnalysisResult` object consumed by the
Streamlit UI, PDF export, DOCX export, and email flow.

## Runtime Flow

Reviewed transcript text flows through the following runtime stages:

1. `ml_mom/transcript_parser.py`
   Parses speaker-labeled transcript text into ordered `TranscriptTurn`
   objects. It validates empty or speakerless transcripts with friendly
   messages instead of raising user-facing exceptions.

2. `ml_mom/preprocessing.py`
   Performs conservative text normalization only: whitespace, punctuation,
   Unicode punctuation, apostrophes, and independent filler words. It preserves
   speaker names, timestamps, dates, numbers, owners, deadlines, and meeting
   terminology.

3. `ml_mom/feature_extraction.py`
   Converts each sentence into a `SentenceFeature` record with lexical,
   structural, meeting-keyword, speaker, and context features. These features
   provide auditable metadata alongside sentence embeddings.

4. `ml_mom/embeddings.py`
   Uses `sentence-transformers` with `all-MiniLM-L6-v2` by default to generate
   local sentence embeddings. The service lazy-loads the model and returns
   friendly errors when the model is unavailable.

5. `ml_mom/clustering.py`
   Clusters sentence embeddings, defaulting to Agglomerative clustering because
   meeting topic counts are usually unknown. Cluster labels are rule-based
   keyword placeholders and do not use LLMs.

6. `ml_mom/predict_ann.py` and `ml_mom/ann_model.py`
   Load `datasets/models/best_model.pt` and
   `datasets/models/label_mapping.json`, then classify each sentence as
   `Discussion`, `Decision`, `Action_Item`, `Summary`, or `Information`.

7. `ml_mom/mom_generator.py`
   Builds structured rule-based Minutes of Meeting from ANN predictions. This
   stage does not call Gemini, OpenAI, Agentic AI, external APIs, or prompt
   templates.

8. `app/main.py`
   Adapts the rule-generated ML minutes back into
   `summarization.base_summarizer.MeetingAnalysisResult`. This adapter keeps the
   existing UI, PDF export, DOCX export, and email workflow unchanged.

## Runtime Inputs

- Reviewed transcript text from audio upload or transcript upload.
- Speaker edits and transcript edits already confirmed by the user.
- Local embedding model availability for `all-MiniLM-L6-v2`.
- Trained ANN artifacts in `datasets/models/`.

## Runtime Outputs

- `MeetingAnalysisResult.cleaned_transcript`
- `MeetingAnalysisResult.summary`
- `MeetingAnalysisResult.key_discussion_points`
- `MeetingAnalysisResult.decisions`
- `MeetingAnalysisResult.action_items`

These outputs intentionally match the existing export contract.

## Training and Dataset Utilities

The following modules support dataset creation and training. They are not part
of production report generation:

- `ml_mom/annotation_tool.py`
- `ml_mom/annotation_app.py`
- `ml_mom/merge_annotations.py`
- `ml_mom/training_dataset.py`
- `ml_mom/train_ann.py`

Annotation CSVs live in `datasets/annotations/`. The merged training dataset is
stored at `datasets/processed/master_dataset.csv`. Trained artifacts are stored
in `datasets/models/`.

## Legacy Areas

The repository still contains legacy Agentic/Gemini files for manual review and
rollback context. Production MoM generation should not import or call them:

- `agentic/`
- `llm_clients/`
- `summarization/llm_summarizer.py`
- `prompts/cleanup/`
- `prompts/summarization/`
- `prompts/extraction/`
- `app/main_old.py`

Do not delete these files automatically. Review them manually after production
verification if cleanup is desired.
