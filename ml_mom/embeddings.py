"""Sentence embedding generation for the future ML MoM pipeline.

Purpose:
    Convert rule-based ``SentenceFeature`` records into numerical sentence
    embeddings that later clustering components can consume.

Responsibilities:
    - Load a local Sentence Transformers model when available.
    - Generate one embedding vector per sentence feature.
    - Preserve sentence, turn, and speaker metadata alongside each vector.
    - Return friendly model-unavailable errors instead of crashing.

Inputs:
    ``SentenceFeature`` objects produced by ``ml_mom.feature_extraction``.

Outputs:
    ``SentenceEmbedding`` objects containing sentence metadata and a numerical
    embedding vector.

Future Implementation Notes:
    The default model is ``all-MiniLM-L6-v2``. The service accepts a model name
    so later phases can swap in another local model without changing callers.
    This module performs no clustering, classification, ANN inference, or
    pipeline integration.
"""

from dataclasses import dataclass, field
import importlib.machinery
import sys
import types
from typing import Any

try:
    from ml_mom.feature_extraction import SentenceFeature
except ModuleNotFoundError:  # pragma: no cover - supports direct script demos.
    from feature_extraction import SentenceFeature


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_UNAVAILABLE_MESSAGE = (
    "Embedding model is unavailable. Please ensure sentence-transformers is "
    "installed and the requested local model can be loaded."
)
_MODEL_CACHE: dict[str, Any] = {}


@dataclass(slots=True)
class SentenceEmbedding:
    """Numerical embedding and metadata for one transcript sentence.

    Attributes:
        sentence_id: Source sentence identifier from ``SentenceFeature``.
        turn_id: Source transcript turn identifier.
        speaker: Normalized speaker label.
        original_sentence: Sentence text used to generate the embedding.
        embedding_vector: Numeric vector returned by the embedding model.
        embedding_dimension: Number of vector dimensions.
    """

    sentence_id: int
    turn_id: int
    speaker: str
    original_sentence: str
    embedding_vector: list[float]
    embedding_dimension: int


@dataclass(slots=True)
class EmbeddingGenerationResult:
    """Container for embedding generation output and friendly error state.

    Attributes:
        embeddings: Generated sentence embeddings.
        error_message: Friendly error message when embeddings could not be
            generated.
        model_name: Name of the embedding model requested by the service.
    """

    embeddings: list[SentenceEmbedding] = field(default_factory=list)
    error_message: str | None = None
    model_name: str = DEFAULT_EMBEDDING_MODEL


class EmbeddingService:
    """Service for loading a sentence-transformers model and creating vectors.

    The service intentionally keeps model loading lazy. This allows the module
    to be imported safely in environments where model files are not yet present.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Sentence Transformers model name to load. The default is
                ``all-MiniLM-L6-v2``.
        """

        self.model_name = model_name
        self._model: Any | None = None
        self.error_message: str | None = None

    def load_model(self) -> str | None:
        """Load the configured Sentence Transformers model.

        Returns:
            ``None`` when the model loads successfully. Otherwise, returns a
            friendly error message describing why embeddings are unavailable.
        """

        if self._model is not None:
            return None

        cached_model = _MODEL_CACHE.get(self.model_name)
        if cached_model is not None:
            self._model = cached_model
            self.error_message = None
            return None

        try:
            _disable_optional_torchcodec_backend()

            # Importing lazily prevents a missing optional dependency from
            # breaking project startup or unrelated modules.
            from sentence_transformers import SentenceTransformer

            # Prefer the cached local model so training remains reliable in
            # restricted/offline environments after the model has been fetched.
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                # If the model has not been cached yet, fall back to the normal
                # loader so a permitted network run can download it once.
                self._model = SentenceTransformer(self.model_name)
            _MODEL_CACHE[self.model_name] = self._model
            self.error_message = None
            return None
        except Exception as exc:  # pragma: no cover - depends on local model state.
            self._model = None
            self.error_message = f"{MODEL_UNAVAILABLE_MESSAGE} Details: {exc}"
            return self.error_message

    def generate_embedding(
        self,
        sentence_feature: SentenceFeature,
    ) -> SentenceEmbedding | None:
        """Generate an embedding for one sentence feature.

        Args:
            sentence_feature: Sentence-level feature record to embed.

        Returns:
            ``SentenceEmbedding`` when the model is available; otherwise
            ``None`` and ``self.error_message`` contains a friendly error.
        """

        load_error = self.load_model()
        if load_error is not None:
            return None

        # The model is checked after loading so static analysis and runtime
        # behavior both know encode is only called on an available model.
        if self._model is None:
            self.error_message = MODEL_UNAVAILABLE_MESSAGE
            return None

        try:
            vector = self._model.encode(
                sentence_feature.original_sentence,
                convert_to_numpy=False,
            )
            embedding_vector = _coerce_vector_to_float_list(vector)
        except Exception as exc:  # pragma: no cover - model-runtime dependent.
            self.error_message = (
                f"Embedding generation failed for sentence "
                f"{sentence_feature.sentence_id}. Details: {exc}"
            )
            return None

        return SentenceEmbedding(
            sentence_id=sentence_feature.sentence_id,
            turn_id=sentence_feature.turn_id,
            speaker=sentence_feature.speaker,
            original_sentence=sentence_feature.original_sentence,
            embedding_vector=embedding_vector,
            embedding_dimension=len(embedding_vector),
        )

    def generate_embeddings(
        self,
        sentence_features: list[SentenceFeature],
    ) -> EmbeddingGenerationResult:
        """Generate embeddings for multiple sentence features.

        Args:
            sentence_features: Sentence-level features to embed.

        Returns:
            ``EmbeddingGenerationResult`` containing generated embeddings or a
            friendly error message when the model is unavailable.
        """

        load_error = self.load_model()
        if load_error is not None:
            return EmbeddingGenerationResult(
                error_message=load_error,
                model_name=self.model_name,
            )

        embeddings: list[SentenceEmbedding] = []
        for sentence_feature in sentence_features:
            embedding = self.generate_embedding(sentence_feature)
            if embedding is None:
                return EmbeddingGenerationResult(
                    embeddings=embeddings,
                    error_message=self.error_message or MODEL_UNAVAILABLE_MESSAGE,
                    model_name=self.model_name,
                )
            embeddings.append(embedding)

        return EmbeddingGenerationResult(
            embeddings=embeddings,
            model_name=self.model_name,
        )

    def get_embedding_dimension(self) -> int | None:
        """Return the embedding dimension for the loaded model.

        Returns:
            Embedding dimension when available; otherwise ``None`` with
            ``self.error_message`` populated.
        """

        load_error = self.load_model()
        if load_error is not None:
            return None

        if self._model is None:
            self.error_message = MODEL_UNAVAILABLE_MESSAGE
            return None

        try:
            if hasattr(self._model, "get_embedding_dimension"):
                return int(self._model.get_embedding_dimension())
            return int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:  # pragma: no cover - model-runtime dependent.
            self.error_message = (
                f"Embedding dimension could not be read. Details: {exc}"
            )
            return None


def _coerce_vector_to_float_list(vector: Any) -> list[float]:
    """Convert model output into a plain Python float list.

    Args:
        vector: Output returned by ``SentenceTransformer.encode``.

    Returns:
        Embedding vector as ``list[float]``.
    """

    # Sentence Transformers may return numpy arrays, tensors, or lists depending
    # on options and installed backends. Plain lists keep downstream modules
    # serialization-friendly and library-neutral.
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _disable_optional_torchcodec_backend() -> None:
    """Disable broken optional TorchCodec detection for text embeddings.

    Sentence-transformers can import Transformers utilities that detect the
    optional ``torchcodec`` audio/video backend. This project uses text-only
    embeddings, so a broken TorchCodec install should not prevent loading
    ``all-MiniLM-L6-v2``.
    """

    try:
        # Sentence-transformers imports modality type hints that reference
        # torchcodec.decoders. The project only embeds text, so a lightweight
        # in-process stub prevents a broken optional DLL from blocking training.
        torchcodec_module = types.ModuleType("torchcodec")
        decoders_module = types.ModuleType("torchcodec.decoders")
        torchcodec_module.__spec__ = importlib.machinery.ModuleSpec(
            name="torchcodec",
            loader=None,
            is_package=True,
        )
        torchcodec_module.__path__ = []
        decoders_module.__spec__ = importlib.machinery.ModuleSpec(
            name="torchcodec.decoders",
            loader=None,
            is_package=False,
        )

        class AudioDecoder:  # pylint: disable=too-few-public-methods
            """Text-only placeholder for optional torchcodec AudioDecoder."""

        class VideoDecoder:  # pylint: disable=too-few-public-methods
            """Text-only placeholder for optional torchcodec VideoDecoder."""

        decoders_module.AudioDecoder = AudioDecoder
        decoders_module.VideoDecoder = VideoDecoder
        torchcodec_module.decoders = decoders_module
        sys.modules["torchcodec"] = torchcodec_module
        sys.modules["torchcodec.decoders"] = decoders_module

        import transformers.utils.import_utils as import_utils

        if hasattr(import_utils.is_torchcodec_available, "cache_clear"):
            import_utils.is_torchcodec_available.cache_clear()
        import_utils.is_torchcodec_available = lambda: False

        try:
            import transformers.utils as transformers_utils

            transformers_utils.is_torchcodec_available = lambda: False
        except Exception:
            pass
    except Exception:
        # This is a best-effort integration guard; if Transformers is not
        # imported yet or changes its internals, normal model loading can still
        # proceed and surface a friendly error through load_model().
        return


def embed_sentence_placeholder(sentence: str) -> list[float]:
    """Backward-compatible helper for the original skeleton entry point.

    Args:
        sentence: Sentence text to embed.

    Returns:
        Embedding vector, or an empty list when the model is unavailable.
    """

    feature = _build_demo_sentence_feature(sentence_id=1, sentence=sentence)
    service = EmbeddingService()
    embedding = service.generate_embedding(feature)
    if embedding is None:
        return []
    return embedding.embedding_vector


def _build_demo_sentence_feature(sentence_id: int, sentence: str) -> SentenceFeature:
    """Build a minimal ``SentenceFeature`` for demos and compatibility helpers.

    Args:
        sentence_id: Demo sentence identifier.
        sentence: Sentence text.

    Returns:
        ``SentenceFeature`` with neutral non-embedding fields.
    """

    return SentenceFeature(
        sentence_id=sentence_id,
        turn_id=sentence_id,
        speaker="Demo Speaker",
        timestamp=None,
        original_sentence=sentence,
        word_count=len(sentence.split()),
        character_count=len(sentence),
        average_word_length=0.0,
        sentence_position=sentence_id,
        is_question=sentence.rstrip().endswith("?"),
        is_exclamation="!" in sentence,
        contains_number=False,
        contains_currency=False,
        contains_percentage=False,
        contains_date=False,
        contains_time=False,
        contains_action_keyword=False,
        contains_decision_keyword=False,
        contains_discussion_keyword=False,
        contains_deadline_keyword=False,
        speaker_name="Demo Speaker",
        speaker_turn_index=1,
        previous_sentence=None,
        next_sentence=None,
    )


def embedding_demo() -> None:
    """Run unit-test style embedding examples.

    Returns:
        None. Prints sentence text, embedding dimension, and the first five
        vector values when the local model is available.
    """

    sentence_features = [
        _build_demo_sentence_feature(
            sentence_id=1,
            sentence="Rahul will prepare the project report by Friday.",
        ),
        _build_demo_sentence_feature(
            sentence_id=2,
            sentence="The team approved the budget today.",
        ),
        _build_demo_sentence_feature(
            sentence_id=3,
            sentence="Priya asked everyone to review the proposal.",
        ),
    ]

    service = EmbeddingService()
    result = service.generate_embeddings(sentence_features)

    if result.error_message is not None:
        print(result.error_message)
        return

    for embedding in result.embeddings:
        print("\nSentence")
        print(embedding.original_sentence)
        print("Embedding Dimension")
        print(embedding.embedding_dimension)
        print("First 5 values")
        print(embedding.embedding_vector[:5])

    # TODO:
    # Add multilingual model selection after multilingual transcript samples are
    # available and evaluated.


if __name__ == "__main__":
    # Windows consoles may use a legacy code page; UTF-8 keeps any future demo
    # examples printable without affecting embedding generation.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    embedding_demo()
