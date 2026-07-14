"""BERT-based metadata extraction helpers for MeetScribe.

Purpose:
    Provide an independent Named Entity Recognition layer for meeting metadata
    such as participants, dates, times, organizations, and locations.

Responsibilities:
    - Keep BERT NER separate from the ANN Minutes of Meeting pipeline.
    - Expose safe metadata extraction functions for the Streamlit runtime.
    - Fall back gracefully when HuggingFace Transformers or model files are
      unavailable.

Inputs:
    Reviewed or newly uploaded transcript text.

Outputs:
    Structured metadata candidates used to prefill meeting information only.

Future Implementation Notes:
    The public runtime entry point is ``extract_meeting_metadata`` from
    ``ml_ner.entity_extractor``. No MoM generation, ANN prediction, clustering,
    embeddings, exports, or UI rendering should be added to this package.
"""

from ml_ner.entity_extractor import EntityExtractionResult, extract_meeting_metadata

__all__ = ["EntityExtractionResult", "extract_meeting_metadata"]
