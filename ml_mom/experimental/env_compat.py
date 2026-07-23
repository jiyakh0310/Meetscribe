"""Experimental environment compatibility shims.

Purpose:
    Keep the experimental formatter runnable in local environments where
    optional native dependencies are unavailable or blocked by Windows
    Application Control.

Scope:
    This module is used only by ``ml_mom.experimental.run_formatter_demo``.
    It does not modify production source files or change production runtime
    behavior.
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


def patch_experimental_embedding_environment(embedding_module: Any | None = None) -> None:
    """Apply experimental-only optional dependency shims.

    Args:
        embedding_module: Optional imported ``ml_mom.embeddings`` module. When
            supplied, its optional TorchCodec guard is patched in-memory for the
            current experimental process only.
    """

    install_torchcodec_stub()
    disable_transformers_torchcodec_detection()

    if embedding_module is not None:
        embedding_module._disable_optional_torchcodec_backend = (  # noqa: SLF001
            patch_experimental_embedding_environment
        )
