"""Abstract `BaseModel` that every quant model in the ecosystem inherits from.

Subclasses must declare three class attributes — `name`, `layer`, `refit_frequency` —
and implement four methods: `fetch_data`, `calibrate`, `predict`, `validate`.
Combined with `@register_model`, this gives the orchestration layer everything
it needs to schedule and route models.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from src.core.data_provider import DataProvider
from src.core.types import CalibrationResult, Forecast, RiskMetric, Signal

RefitFrequency = Literal["tick", "intraday", "daily", "weekly", "monthly"]

_VALID_REFIT_FREQUENCIES: frozenset[str] = frozenset(
    {"tick", "intraday", "daily", "weekly", "monthly"}
)


class BaseModel(ABC):
    """Abstract base for every model. See module docstring for the contract."""

    name: ClassVar[str]
    layer: ClassVar[int]
    refit_frequency: ClassVar[RefitFrequency]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        missing = [
            attr
            for attr in ("name", "layer", "refit_frequency")
            if attr not in cls.__dict__ and not _attr_inherited_from_concrete(cls, attr)
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define class attributes: {missing}"
            )
        if cls.refit_frequency not in _VALID_REFIT_FREQUENCIES:
            raise ValueError(
                f"{cls.__name__}.refit_frequency must be one of "
                f"{sorted(_VALID_REFIT_FREQUENCIES)}, got {cls.refit_frequency!r}"
            )
        if not isinstance(cls.layer, int):
            raise TypeError(
                f"{cls.__name__}.layer must be int, got {type(cls.layer).__name__}"
            )

    @abstractmethod
    def fetch_data(self, provider: DataProvider) -> Any:
        """Pull every input this model needs from `provider`."""

    @abstractmethod
    def calibrate(self, data: Any) -> CalibrationResult:
        """Fit / refit the model. Called at the cadence given by `refit_frequency`."""

    @abstractmethod
    def predict(self, data: Any) -> Signal | Forecast | RiskMetric:
        """Produce a single model output from already-fetched data."""

    @abstractmethod
    def validate(self, data: Any) -> dict[str, Any]:
        """Diagnostic checks (residuals, drift, data quality). Free-form keys."""


def _attr_inherited_from_concrete(cls: type, attr: str) -> bool:
    """True iff `attr` is set on some non-`BaseModel` ancestor of `cls`."""
    for base in cls.__mro__[1:]:
        if base is BaseModel or base is object:
            continue
        if attr in base.__dict__:
            return True
    return False
