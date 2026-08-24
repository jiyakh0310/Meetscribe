"""Configuration loading and path resolution for Voxels."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from voxels.labels import CANONICAL_LABELS

CONFIG_FILENAMES = ("paths.yaml", "audio.yaml", "model.yaml", "training.yaml")


def project_root() -> Path:
    """Return the repository root inferred from the installed source tree."""
    return Path(__file__).resolve().parents[2]


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mapping values into a new dictionary."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load one YAML file and return a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data


def load_project_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Load and merge the project's YAML configuration files."""
    base_dir = Path(config_dir) if config_dir is not None else project_root() / "configs"
    config: dict[str, Any] = {}
    for filename in CONFIG_FILENAMES:
        config = deep_merge(config, load_yaml_file(base_dir / filename))
    config.setdefault("model", {})
    config["model"]["labels"] = list(CANONICAL_LABELS)
    config["model"]["num_labels"] = len(CANONICAL_LABELS)
    return config


def resolve_project_path(relative_path: str | Path, root: Path | None = None) -> Path:
    """Resolve a project-relative path without embedding machine-specific paths."""
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Expected a project-relative path, got absolute path: {path}")
    return (root or project_root()).joinpath(path).resolve()


def resolved_paths(config: Mapping[str, Any] | None = None, root: Path | None = None) -> dict[str, Path]:
    """Return configured project paths resolved against the repository root."""
    active_config = config or load_project_config()
    paths = active_config.get("paths", {})
    if not isinstance(paths, Mapping):
        raise ValueError("The 'paths' configuration section must be a mapping.")
    return {key: resolve_project_path(Path(value), root=root) for key, value in paths.items()}
