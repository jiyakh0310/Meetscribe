"""Google Gemini client adapter for the Phase 2 summarization pipeline."""

from __future__ import annotations

from config.settings import SettingsError, get_settings

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClientError(RuntimeError):
    """Raised when Gemini client setup or generation fails."""


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
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=self._temperature,
                system_instruction=system_prompt,
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise GeminiClientError(f"Gemini generation failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise GeminiClientError("Gemini returned an empty response.")

        return text.strip()
