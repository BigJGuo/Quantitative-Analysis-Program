"""Shared core infrastructure for the multi-agent quantitative analysis ecosystem.

Each model lives under `src/models/<model_name>/` and depends on this package.
The core is read-only from a model's perspective — see CONTRIBUTING.md.
"""

from __future__ import annotations

from src.core.base_model import BaseModel, RefitFrequency
from src.core.config import CoreConfig, load_config
from src.core.data_provider import (
    CacheConfig,
    DataProvider,
    DiskCache,
    InMemoryProvider,
    YFinanceProvider,
)
from src.core.registry import (
    clear_registry,
    get_model,
    list_models,
    models_by_layer,
    register_model,
)
from src.core.types import (
    CalibrationResult,
    Forecast,
    RiskMetric,
    Signal,
    SignalDirection,
)

__all__ = [
    "BaseModel",
    "CacheConfig",
    "CalibrationResult",
    "CoreConfig",
    "DataProvider",
    "DiskCache",
    "Forecast",
    "InMemoryProvider",
    "RefitFrequency",
    "RiskMetric",
    "Signal",
    "SignalDirection",
    "YFinanceProvider",
    "clear_registry",
    "get_model",
    "list_models",
    "load_config",
    "models_by_layer",
    "register_model",
]
