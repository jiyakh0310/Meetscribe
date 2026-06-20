"""Application configuration loaded from Streamlit secrets, env, or .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

REQUIRED_ENV_VARS: tuple[str, ...] = ("SARVAM_API_KEY",)
OPTIONAL_ENV_VARS: tuple[str, ...] = ("GEMINI_API_KEY",)


class SettingsError(ValueError):
    """Raised when required configuration is missing or invalid."""


def _masked(value: str) -> str:
    return f"{value[:6]}..."


def _debug_loaded(name: str, source: str, value: str | None) -> None:
    if value:
        print(f"Loaded {name} from {source}: {_masked(value)}")
    else:
        print(f"{name} not loaded from {source}.")


def _read_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        value = str(st.secrets[name]).strip()
    except Exception:
        value = ""

    result = value or None
    _debug_loaded(name, "Streamlit secrets", result)
    return result


def _read_environment_variable(name: str) -> str | None:
    value = os.getenv(name, "").strip() or None
    _debug_loaded(name, "environment variables", value)
    return value


def _read_dotenv_variable(name: str) -> str | None:
    dotenv_data = dotenv_values(ENV_FILE)
    value = str(dotenv_data.get(name, "") or "").strip() or None
    _debug_loaded(name, ".env", value)
    return value


def _read_config_value(name: str) -> str | None:
    streamlit_secret = _read_streamlit_secret(name)
    environment_variable = _read_environment_variable(name)
    dotenv_variable = _read_dotenv_variable(name)
    return streamlit_secret or environment_variable or dotenv_variable


def _read_required_settings() -> dict[str, str]:
    """Return validated required variables from secrets, env, or .env."""

    missing: list[str] = []
    values: dict[str, str] = {}

    for name in REQUIRED_ENV_VARS:
        value = _read_config_value(name)
        if not value:
            missing.append(name)
        else:
            values[name] = value

    if missing:
        missing_list = ", ".join(missing)
        raise SettingsError(
            f"Missing required configuration value(s): {missing_list}. "
            "Define them in Streamlit secrets, environment variables, or .env."
        )

    return values


def _read_optional_settings() -> dict[str, str | None]:
    """Return optional variables from secrets, env, or .env when present."""
    return {name: _read_config_value(name) for name in OPTIONAL_ENV_VARS}


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings."""

    sarvam_api_key: str
    gemini_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        env = _read_required_settings()
        optional_env = _read_optional_settings()
        return cls(
            sarvam_api_key=env["SARVAM_API_KEY"],
            gemini_api_key=optional_env["GEMINI_API_KEY"],
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance (loads and validates on first call)."""
    return Settings.from_env()
