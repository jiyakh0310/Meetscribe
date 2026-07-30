"""Experimental environment compatibility shims.

Purpose:
    Keep the experimental formatter runnable in local environments where
    optional native dependencies are unavailable or blocked by Windows
    Application Control.

Scope:
    This module is used by the experimental formatter integration path. It does
    not modify model files, training logic, ANN inference, MiniLM weights, or
    the deterministic formatter.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any


def install_torchcodec_stub() -> None:
    """Install a spec-bearing TorchCodec stub for text-only MiniLM usage.

    Sentence embedding for the experimental formatter is text-only. Some
    versions of ``transformers`` probe optional ``torchcodec`` support during
    import/model loading. When ``torchcodec`` is missing or blocked, a minimal
    module with a valid ``__spec__`` prevents optional-backend detection from
    failing before MiniLM can load.
    """

    torchcodec_module = types.ModuleType("torchcodec")
    torchcodec_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchcodec",
        loader=None,
        is_package=True,
    )
    torchcodec_module.__path__ = []

    decoders_module = types.ModuleType("torchcodec.decoders")
    decoders_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchcodec.decoders",
        loader=None,
        is_package=False,
    )

    class AudioDecoder:  # pylint: disable=too-few-public-methods
        """Placeholder for optional TorchCodec audio decoding."""

    class VideoDecoder:  # pylint: disable=too-few-public-methods
        """Placeholder for optional TorchCodec video decoding."""

    decoders_module.AudioDecoder = AudioDecoder
    decoders_module.VideoDecoder = VideoDecoder
    torchcodec_module.decoders = decoders_module

    sys.modules["torchcodec"] = torchcodec_module
    sys.modules["torchcodec.decoders"] = decoders_module


def disable_transformers_torchcodec_detection() -> None:
    """Tell Transformers that optional TorchCodec support is unavailable."""

    try:
        import transformers.utils.import_utils as import_utils

        if hasattr(import_utils.is_torchcodec_available, "cache_clear"):
            import_utils.is_torchcodec_available.cache_clear()
        import_utils.is_torchcodec_available = lambda: False
    except Exception:
        pass

    try:
        import transformers.utils as transformers_utils

        transformers_utils.is_torchcodec_available = lambda: False
    except Exception:
        pass


def install_datasets_stub() -> None:
    """Install a minimal HuggingFace Datasets stub for inference-only MiniLM.

    ``sentence-transformers`` imports ``datasets`` while building model-card
    metadata. That optional path imports pandas native extensions, which can be
    blocked by Windows Application Control in locked-down environments. Runtime
    sentence embedding does not need dataset loading, so this stub provides only
    the names imported by ``sentence_transformers.model_card`` and prevents the
    blocked pandas DLL from being loaded.
    """

    datasets_module = types.ModuleType("datasets")
    datasets_module.__version__ = "0.0.0-optional-stub"
    datasets_module.__spec__ = importlib.machinery.ModuleSpec(
        name="datasets",
        loader=None,
        is_package=True,
    )
    datasets_module.__path__ = []

    class Dataset:  # pylint: disable=too-few-public-methods
        """Placeholder for optional HuggingFace Dataset metadata."""

    class DatasetDict(dict):
        """Placeholder for optional HuggingFace DatasetDict metadata."""

    class Value:  # pylint: disable=too-few-public-methods
        """Placeholder for optional HuggingFace feature value metadata."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    datasets_module.Dataset = Dataset
    datasets_module.DatasetDict = DatasetDict
    datasets_module.Value = Value
    sys.modules["datasets"] = datasets_module


def patch_experimental_embedding_environment(embedding_module: Any | None = None) -> None:
    """Apply experimental-only optional dependency shims.

    Args:
        embedding_module: Optional imported ``ml_mom.embeddings`` module. When
            supplied, its optional TorchCodec guard is patched in-memory for the
            current experimental process only.
    """

    install_torchcodec_stub()
    install_datasets_stub()
    disable_transformers_torchcodec_detection()

    if embedding_module is not None:
        embedding_module._disable_optional_torchcodec_backend = (  # noqa: SLF001
            patch_experimental_embedding_environment
        )
