"""BERT Named Entity Recognition service for meeting metadata.

Purpose:
    Run a HuggingFace Transformer token-classification model over transcript
    text and return BIO-tagged entity spans.

Responsibilities:
    - Load ``dslim/bert-base-NER`` with its matching WordPiece tokenizer.
    - Cache the tokenizer/model pipeline so repeated uploads do not reload BERT.
    - Convert model output into internal BIO records.
    - Return friendly error messages instead of raising runtime exceptions.

Inputs:
    Raw transcript text or speaker-labeled transcript text.

Outputs:
    ``BIONerEntity`` objects containing entity text, BIO labels, confidence,
    and character offsets.

Future Implementation Notes:
    This module is metadata-only. It must not call the ANN model, embeddings,
    clustering, rule-based MoM generation, exports, email, or UI components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
import time
from typing import Any


DEFAULT_NER_MODEL = "dslim/bert-base-NER"
SUPPORTED_ENTITY_GROUPS = {"PER", "DATE", "TIME", "ORG", "LOC"}
MODEL_UNAVAILABLE_MESSAGE = (
    "BERT NER model is unavailable. Falling back to existing metadata extraction."
)
_MODEL_LOAD_FAILURES: dict[str, str] = {}


@dataclass(slots=True)
class BIONerToken:
    """One token-level BIO prediction from the BERT NER model.

    BIO notation labels the beginning and inside of an entity span. For example,
    ``Rahul Sharma`` should become ``B-PER`` for ``Rahul`` and ``I-PER`` for
    ``Sharma``. WordPiece subword fragments are retained internally for audit
    but are not exposed in the UI.

    Attributes:
        token: Token text returned by the HuggingFace tokenizer.
        bio_tag: BIO label such as ``B-PER`` or ``I-ORG``.
        entity_group: Entity group without the BIO prefix.
        confidence: Model probability for this token.
        start: Character start offset in the input text.
        end: Character end offset in the input text.
    """

    token: str
    bio_tag: str
    entity_group: str
    confidence: float
    start: int | None = None
    end: int | None = None


@dataclass(slots=True)
class BIONerEntity:
    """Merged entity span derived from token-level BIO predictions.

    Attributes:
        text: Human-readable entity text reconstructed from model offsets.
        entity_group: Normalized entity group such as ``PER`` or ``ORG``.
        bio_tags: Token-level BIO tags that formed this entity.
        confidence: Average confidence across entity tokens.
        start: Character start offset in the input text.
        end: Character end offset in the input text.
        tokens: Internal WordPiece/BIO token predictions for audit logging.
    """

    text: str
    entity_group: str
    bio_tags: list[str]
    confidence: float
    start: int | None = None
    end: int | None = None
    tokens: list[BIONerToken] = field(default_factory=list)


@dataclass(slots=True)
class NerInferenceResult:
    """NER inference output and non-fatal error state.

    Attributes:
        entities: Merged entity spans detected in the text.
        tokens: Token-level BIO predictions retained internally.
        model_name: HuggingFace model identifier.
        inference_time_seconds: Runtime duration for model inference.
        error_message: Friendly message when model loading or inference fails.
    """

    entities: list[BIONerEntity] = field(default_factory=list)
    tokens: list[BIONerToken] = field(default_factory=list)
    model_name: str = DEFAULT_NER_MODEL
    inference_time_seconds: float = 0.0
    error_message: str | None = None


class BertNerService:
    """Cached HuggingFace BERT NER service.

    The model and tokenizer are loaded through ``transformers.pipeline`` using
    the model's own WordPiece tokenizer. Loading is cached globally through
    ``_load_token_classifier`` so production requests reuse the same model.
    """

    def __init__(self, model_name: str = DEFAULT_NER_MODEL) -> None:
        """Initialize the NER service.

        Args:
            model_name: HuggingFace token-classification model name. The
                default is ``dslim/bert-base-NER``.
        """

        self.model_name = model_name

    def extract_entities(self, transcript_text: str) -> NerInferenceResult:
        """Run BERT NER over transcript text.

        Args:
            transcript_text: Raw or speaker-labeled transcript text.

        Returns:
            ``NerInferenceResult`` with BIO entities, or an error message when
            the model is unavailable. Exceptions are intentionally converted
            into friendly fallback states because metadata extraction must never
            block upload or MoM generation.
        """

        if not transcript_text or not transcript_text.strip():
            return NerInferenceResult(model_name=self.model_name)

        started_at = time.perf_counter()
        cached_failure = _MODEL_LOAD_FAILURES.get(self.model_name)
        if cached_failure:
            return NerInferenceResult(
                model_name=self.model_name,
                inference_time_seconds=time.perf_counter() - started_at,
                error_message=cached_failure,
            )
        if not _model_download_allowed() and not _has_local_huggingface_model(self.model_name):
            error_message = (
                f"{MODEL_UNAVAILABLE_MESSAGE} Details: local HuggingFace cache "
                f"does not contain '{self.model_name}'."
            )
            _MODEL_LOAD_FAILURES[self.model_name] = error_message
            return NerInferenceResult(
                model_name=self.model_name,
                inference_time_seconds=time.perf_counter() - started_at,
                error_message=error_message,
            )

        try:
            classifier = _load_token_classifier(self.model_name)

            # The selected BERT model uses WordPiece tokenization internally.
            # We ask HuggingFace for non-aggregated token predictions so the
            # application can preserve BIO tags before resolving full entities.
            raw_tokens = classifier(
                transcript_text,
                aggregation_strategy="none",
                truncation=True,
            )
            tokens = [_token_from_prediction(prediction) for prediction in raw_tokens]
            entities = merge_bio_tokens(tokens, transcript_text)
            return NerInferenceResult(
                entities=entities,
                tokens=tokens,
                model_name=self.model_name,
                inference_time_seconds=time.perf_counter() - started_at,
            )
        except Exception as exc:  # pragma: no cover - depends on local model state.
            error_message = f"{MODEL_UNAVAILABLE_MESSAGE} Details: {exc}"
            _MODEL_LOAD_FAILURES[self.model_name] = error_message
            return NerInferenceResult(
                model_name=self.model_name,
                inference_time_seconds=time.perf_counter() - started_at,
                error_message=error_message,
            )


@lru_cache(maxsize=2)
def _load_token_classifier(model_name: str) -> Any:
    """Load and cache the HuggingFace token-classification pipeline.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A Transformers token-classification pipeline.
    """

    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    # The tokenizer must come from the same model family so WordPiece offsets
    # and BIO tags align with the trained BERT NER checkpoint.
    allow_download = _model_download_allowed()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=not allow_download,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        local_files_only=not allow_download,
    )
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
    )


def _model_download_allowed() -> bool:
    """Return whether runtime may download HuggingFace NER model files.

    Returns:
        ``True`` only when ``MEETSCRIBE_NER_ALLOW_DOWNLOAD=1`` is set.
    """

    return os.getenv("MEETSCRIBE_NER_ALLOW_DOWNLOAD", "").strip() == "1"


def _has_local_huggingface_model(model_name: str) -> bool:
    """Check whether a HuggingFace model appears to be cached locally.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        ``True`` when a snapshot with ``config.json`` exists in a common cache
        location. This avoids slow network/cache probing during Streamlit
        upload interactions when the model is unavailable.
    """

    cache_roots = []
    hf_home = os.getenv("HF_HOME")
    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")
    if transformers_cache:
        cache_roots.append(Path(transformers_cache))
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    model_cache_name = "models--" + model_name.replace("/", "--")
    for cache_root in cache_roots:
        model_root = cache_root / model_cache_name
        snapshots = model_root / "snapshots"
        if not snapshots.is_dir():
            continue
        if any(snapshot.joinpath("config.json").is_file() for snapshot in snapshots.iterdir()):
            return True
    return False


def _token_from_prediction(prediction: dict[str, Any]) -> BIONerToken:
    """Convert a HuggingFace token prediction into a BIO token record.

    Args:
        prediction: Raw prediction dictionary returned by Transformers.

    Returns:
        ``BIONerToken`` normalized for downstream entity resolution.
    """

    bio_tag = str(prediction.get("entity") or "O")
    entity_group = bio_tag.split("-", 1)[-1] if "-" in bio_tag else bio_tag
    return BIONerToken(
        token=str(prediction.get("word") or ""),
        bio_tag=bio_tag,
        entity_group=entity_group,
        confidence=float(prediction.get("score") or 0.0),
        start=prediction.get("start"),
        end=prediction.get("end"),
    )


def merge_bio_tokens(tokens: list[BIONerToken], source_text: str) -> list[BIONerEntity]:
    """Merge token-level BIO predictions into entity spans.

    Args:
        tokens: Token-level BIO predictions.
        source_text: Original transcript text used for character offsets.

    Returns:
        Ordered merged entity spans.
    """

    entities: list[BIONerEntity] = []
    current_tokens: list[BIONerToken] = []
    current_group = ""

    def flush_current() -> None:
        if not current_tokens:
            return
        entity = _entity_from_tokens(current_tokens, current_group, source_text)
        if entity is not None:
            entities.append(entity)

    for token in tokens:
        group = token.entity_group
        if token.bio_tag == "O" or group not in SUPPORTED_ENTITY_GROUPS:
            flush_current()
            current_tokens = []
            current_group = ""
            continue

        is_begin = token.bio_tag.startswith("B-")
        group_changed = current_group and group != current_group
        if is_begin or group_changed:
            flush_current()
            current_tokens = [token]
            current_group = group
            continue

        if not current_tokens:
            current_tokens = [token]
            current_group = group
        else:
            current_tokens.append(token)

    flush_current()
    return entities


def _entity_from_tokens(
    tokens: list[BIONerToken],
    entity_group: str,
    source_text: str,
) -> BIONerEntity | None:
    """Build a merged entity from contiguous BIO tokens.

    Args:
        tokens: BIO tokens belonging to one entity.
        entity_group: Entity group shared by the tokens.
        source_text: Original transcript text for offset reconstruction.

    Returns:
        Merged entity, or ``None`` if no text can be reconstructed.
    """

    starts = [token.start for token in tokens if token.start is not None]
    ends = [token.end for token in tokens if token.end is not None]
    start = min(starts) if starts else None
    end = max(ends) if ends else None

    if start is not None and end is not None and 0 <= start < end <= len(source_text):
        text = source_text[start:end]
    else:
        # Offset reconstruction is preferred because WordPiece tokens may look
        # like "Rah", "##ul"; this fallback keeps degraded model outputs usable.
        text = " ".join(token.token.replace("##", "") for token in tokens)

    text = " ".join(text.split()).strip()
    if not text:
        return None

    confidence = sum(token.confidence for token in tokens) / len(tokens)
    return BIONerEntity(
        text=text,
        entity_group=entity_group,
        bio_tags=[token.bio_tag for token in tokens],
        confidence=round(confidence, 6),
        start=start,
        end=end,
        tokens=list(tokens),
    )
