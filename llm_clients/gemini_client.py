"""Google Gemini client adapter for the Phase 2 summarization pipeline."""

from __future__ import annotations

import logging
import time

from config.settings import SettingsError, get_settings

DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_HIGH_DEMAND_MESSAGE = (
    "Transcript was generated successfully, but meeting analysis is temporarily "
    "unavailable due to high Gemini API demand. Please try again in a few minutes."
)
RETRY_DELAYS_SECONDS = (2, 5, 10)
logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """Raised when Gemini client setup or generation fails."""


def _gemini_status_code(exc: Exception) -> int | None:
    for attr_name in ("code", "status_code"):
        value = getattr(exc, attr_name, None)
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 503 if "503" in str(exc) else None


def _is_gemini_unavailable(exc: Exception) -> bool:
    status_code = _gemini_status_code(exc)
    message = str(exc).upper()
    return status_code == 503 or "UNAVAILABLE" in message


class GeminiClient:
    """Adapter matching LLMSummarizer's generate(prompt, system_prompt) protocol."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
    ) -> None:
        self._api_key = api_key or self._load_api_key()
        self._model = model
        self._temperature = temperature

        if not self._api_key:
            raise GeminiClientError(
                "Missing GEMINI_API_KEY. Define it in Streamlit secrets, "
                "environment variables, .env, or pass api_key."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise GeminiClientError(
                "Missing google-genai package. Install dependencies with "
                "'pip install -r requirements.txt'."
            ) from exc

        self._client = genai.Client(api_key=self._api_key)

    def _load_api_key(self) -> str:
        try:
            settings = get_settings()
        except SettingsError as exc:
            raise GeminiClientError(str(exc)) from exc

        return settings.gemini_api_key or ""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            system_instruction=system_prompt,
        )

        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                break
            except Exception as exc:
                status_code = _gemini_status_code(exc)
                if not _is_gemini_unavailable(exc):
                    raise GeminiClientError(f"Gemini generation failed: {exc}") from exc

                if attempt >= len(RETRY_DELAYS_SECONDS):
                    logger.error(
                        "Gemini unavailable after %d retries: status_code=%s reason=%s",
                        len(RETRY_DELAYS_SECONDS),
                        status_code,
                        exc,
                    )
                    raise GeminiClientError(GEMINI_HIGH_DEMAND_MESSAGE) from exc

                retry_number = attempt + 1
                delay_seconds = RETRY_DELAYS_SECONDS[attempt]
                logger.warning(
                    "Gemini unavailable; retry attempt %d after %s seconds "
                    "(status_code=%s)",
                    retry_number,
                    delay_seconds,
                    status_code,
                )
                time.sleep(delay_seconds)

        text = getattr(response, "text", None)
        if not text:
            raise GeminiClientError("Gemini returned an empty response.")

        return text.strip()
