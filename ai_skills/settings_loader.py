#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for loading project settings from TOML files.

Configuration now lives under ``config/`` so instructors can override values
without editing Python source files. Values flow through in this order:

1. ``config/settings.toml`` (checked into the repo, holds shared defaults)
2. ``config/settings.local.toml`` (optional, git-ignored overrides)
3. Environment variables (handled in ``ai_skills.config``)

Both TOML files use friendly, sectioned keys, but the rest of the application
still expects the historical ``UPPER_SNAKE_CASE`` names. This module normalizes
the TOML entries back into that format and expands relative paths to absolute
ones rooted at the project directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_DIR_NAME = "config"
BASE_SETTINGS_NAME = "settings.toml"
LOCAL_SETTINGS_NAME = "settings.local.toml"

_PATH_KEYS = {
    "DATA_DIR",
    "INPUTS_DIR",
    "OUTPUTS_DIR",
    "INPUT_CSV",
    "OUTPUT_CSV",
}

_SECTION_KEY_MAPPING: dict[str, dict[str, str]] = {
    "paths": {
        "data_dir": "DATA_DIR",
        "inputs_dir": "INPUTS_DIR",
        "outputs_dir": "OUTPUTS_DIR",
        "input_csv": "INPUT_CSV",
        "output_csv": "OUTPUT_CSV",
    },
    "openai": {
        "model": "OPENAI_MODEL",
        "temperature": "OPENAI_TEMPERATURE",
        "rate_limit_delay": "RATE_LIMIT_DELAY",
        "batch_size": "OPENAI_BATCH_SIZE",
        "max_parallel_requests": "OPENAI_MAX_PARALLEL_REQUESTS",
        "service_tier": "SERVICE_TIER",
        "timeout": "OPENAI_TIMEOUT",
        "max_retries": "OPENAI_MAX_RETRIES",
    },
    "processing": {"max_job_desc_length": "MAX_JOB_DESC_LENGTH"},
}


def load_user_config(project_root: Path) -> Dict[str, Any]:
    """Return a flat dictionary of overrides derived from TOML settings."""
    config_dir = project_root / CONFIG_DIR_NAME
    config: dict[str, Any] = {}
    for filename in (BASE_SETTINGS_NAME, LOCAL_SETTINGS_NAME):
        config.update(_load_single_file(config_dir / filename, project_root))
    return config


def _load_single_file(path: Path, project_root: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file at {path} must contain key/value pairs.")
    return _flatten_sections(data, project_root, source=path)


def _flatten_sections(
    data: Mapping[str, Any],
    project_root: Path,
    *,
    source: Path,
) -> Dict[str, Any]:
    flattened: dict[str, Any] = {}
    for section_name, value in data.items():
        if section_name in _SECTION_KEY_MAPPING:
            flattened.update(
                _flatten_known_section(
                    section_name, value, project_root=project_root, source=source
                )
            )
        else:
            normalized_key = section_name.upper()
            flattened[normalized_key] = _normalize_value(
                normalized_key, value, project_root
            )
    return flattened


def _flatten_known_section(
    section: str,
    values: Any,
    *,
    project_root: Path,
    source: Path,
) -> Dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(
            f"Section [{section}] in {source} must contain nested key/value pairs."
        )
    mapping = _SECTION_KEY_MAPPING[section]
    flattened: dict[str, Any] = {}
    for key, env_name in mapping.items():
        if key in values:
            flattened[env_name] = _normalize_value(
                env_name, values[key], project_root
            )
    return flattened


def _normalize_value(key: str, value: Any, project_root: Path) -> Any:
    if key in _PATH_KEYS and isinstance(value, str):
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    return value


__all__ = ["load_user_config"]
