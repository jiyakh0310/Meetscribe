"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

REQUIRED_ENV_VARS: tuple[str, ...] = ("SARVAM_API_KEY",)
OPTIONAL_ENV_VARS: tuple[str, ...] = ("GEMINI_API_KEY",)


class SettingsError(ValueError):
    """Raised when required configuration is missing or invalid."""


def _load_dotenv() -> None:
    load_dotenv(ENV_FILE, override=False)


def _read_required_env() -> dict[str, str]:
    """Load .env and return validated required variables."""
    _load_dotenv()

    missing: list[str] = []
    values: dict[str, str] = {}

    for name in REQUIRED_ENV_VARS:
        value = os.getenv(name, "").strip()
        if not value:
            missing.append(name)
        else:
            values[name] = value

    if missing:
        missing_list = ", ".join(missing)
        raise SettingsError(
            f"Missing required environment variable(s): {missing_list}. "
            f"Define them in {ENV_FILE} or export them in your shell."
        )

    return values


def _read_optional_env() -> dict[str, str | None]:
    """Load .env and return optional variables when present."""
    _load_dotenv()
    return {
        name: os.getenv(name, "").strip() or None
        for name in OPTIONAL_ENV_VARS
    }


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings."""

    sarvam_api_key: str
    gemini_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        env = _read_required_env()
        optional_env = _read_optional_env()
        return cls(
            sarvam_api_key=env["SARVAM_API_KEY"],
            gemini_api_key=optional_env["GEMINI_API_KEY"],
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance (loads and validates on first call)."""
    return Settings.from_env()
