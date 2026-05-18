"""Shared result dataclasses returned by model `predict` / `calibrate` methods.

These types are deliberately broad so that the 20 downstream models — spanning
structural, ML signal, execution, and risk layers — can return values without
having to define their own ad-hoc shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SignalDirection = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class Signal:
    """Directional trading signal for a single instrument."""

    ticker: str
    direction: SignalDirection
    strength: float
    timestamp: datetime
    horizon: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Signal.strength must be in [0, 1], got {self.strength}"
            )


@dataclass(frozen=True)
class Forecast:
    """Point forecast with optional prediction interval."""

    ticker: str
    horizon: str
    value: float
    timestamp: datetime
    lower: float | None = None
    upper: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(
                f"Forecast.lower ({self.lower}) must not exceed upper ({self.upper})"
            )


@dataclass(frozen=True)
class RiskMetric:
    """Single-value risk metric (VaR, ES, volatility, beta, ...)."""

    ticker: str
    metric_name: str
    value: float
    timestamp: datetime
    confidence_level: float | None = None
    horizon: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationResult:
    """Output of `BaseModel.calibrate` — fit parameters plus diagnostics."""

    model_name: str
    parameters: dict[str, Any]
    fit_metrics: dict[str, float]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
