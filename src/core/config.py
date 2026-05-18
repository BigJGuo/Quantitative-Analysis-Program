"""TOML-backed configuration for the core.

Layout of `config.toml` (all sections optional):

    [core]
    cache_dir = ".cache/data"
    default_ttl_seconds = 3600

    [core.per_method_ttl_seconds]
    fetch_fundamentals = 86400
    fetch_intraday     = 300

    [models.my_model]
    ...  # picked up by individual models via CoreConfig.extras

Models read their own section out of `CoreConfig.extras`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.data_provider import CacheConfig


@dataclass
class CoreConfig:
    """Top-level configuration consumed by the core."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    extras: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path = "config.toml") -> CoreConfig:
    """Load `path` if it exists; otherwise return defaults."""
    p = Path(path)
    if not p.exists():
        return CoreConfig()
    with p.open("rb") as fh:
        raw = tomllib.load(fh)
    core_section = raw.get("core", {})
    cache = CacheConfig(
        cache_dir=Path(core_section.get("cache_dir", ".cache/data")),
        default_ttl_seconds=int(core_section.get("default_ttl_seconds", 3600)),
        per_method_ttl_seconds=dict(core_section.get("per_method_ttl_seconds", {})),
    )
    extras = {k: v for k, v in raw.items() if k != "core"}
    return CoreConfig(cache=cache, extras=extras)
