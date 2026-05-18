"""Dataclasses specific to the Kyle-lambda / square-root impact model.

Two distinct estimators are exposed and they have different parameter
shapes:

* ``KyleLambdaFit`` — output of the linear OLS regression
  ``r_t = lambda * Q_t^signed + epsilon_t``. Carries the point estimate,
  HC1 standard error, 95% confidence interval, R^2 and observation count.
* ``SqrtImpactPrediction`` — the pre-trade cost forecast from the
  square-root law ``Delta P / P = Y * sigma * sqrt(Q / V)``. Holds the
  bps and dollar values, plus the crossover with the linear-Kyle estimate.

Shared shapes (``Signal``, ``Forecast``, ``RiskMetric``, ``CalibrationResult``)
live in ``src.core.types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd

KyleFrequency = Literal["daily", "intraday"]
VALID_FREQUENCIES: frozenset[str] = frozenset({"daily", "intraday"})


@dataclass(frozen=True)
class KyleLambdaFit:
    """Output of the Kyle-lambda regression.

    Attributes
    ----------
    lambda_hat:
        OLS point estimate of the per-share return-on-signed-volume slope.
        Units: return per share. For dollar-signed volume the units become
        return per dollar.
    se:
        HC1 (heteroskedasticity-robust) standard error of ``lambda_hat``.
    ci_low / ci_high:
        95% Wald confidence interval ``lambda_hat +/- 1.96 * se``.
    r_squared:
        Goodness-of-fit of the no-intercept regression.
    n_obs:
        Number of observations used in the regression.
    frequency:
        ``"daily"`` or ``"intraday"`` — drives how the inputs were assembled.
    mean_signed_volume_abs:
        Mean of ``|Q_t^signed|`` — useful for normalizing ``lambda`` to a
        dimensionless "return per typical bar of one-sided flow".
    mean_volume:
        Mean of unsigned bar volume; the standard normalization is
        ``lambda * mean_volume`` ("return per one-day's volume one-sided").
    """

    lambda_hat: float
    se: float
    ci_low: float
    ci_high: float
    r_squared: float
    n_obs: int
    frequency: KyleFrequency
    mean_signed_volume_abs: float
    mean_volume: float

    def __post_init__(self) -> None:
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"KyleLambdaFit.frequency must be one of "
                f"{sorted(VALID_FREQUENCIES)}, got {self.frequency!r}"
            )
        if self.n_obs < 2:
            raise ValueError(f"KyleLambdaFit.n_obs must be >= 2, got {self.n_obs}")
        if self.se < 0.0:
            raise ValueError(f"KyleLambdaFit.se must be >= 0, got {self.se}")
        if self.ci_low > self.ci_high:
            raise ValueError(
                f"ci_low ({self.ci_low}) must not exceed ci_high ({self.ci_high})"
            )

    @property
    def lambda_normalized(self) -> float:
        """``lambda * mean_volume`` — the return produced by one bar's worth
        of one-sided flow at the per-bar mean. For daily data this is the
        spec's "return per one daily-volume" quantity.
        """

        return self.lambda_hat * self.mean_volume


@dataclass(frozen=True)
class SqrtImpactPrediction:
    """Output of Algorithm C — square-root pre-trade impact.

    Attributes
    ----------
    impact_relative:
        Signed relative price drift ``Delta P / P``. Sign is ``+1`` for a
        buy metaorder, ``-1`` for a sell.
    impact_bps:
        ``1e4 * |impact_relative|`` — the canonical units for execution
        reporting.
    impact_dollars:
        Signed dollar cost ``side * impact_relative * P * Q``.
    sigma_daily:
        The daily volatility used as the ``sigma`` input.
    volume_daily:
        The reference daily volume used as the ``V`` input.
    Q:
        Parent order size in shares.
    Q_over_V:
        Participation ratio ``Q / V``. Exceeding ``0.10`` triggers the spec
        warning (the law is calibrated only up to that ratio).
    Y_prefactor:
        Dimensionless prefactor used (literature prior unless a fit is
        supplied).
    extrapolation_warning:
        ``True`` whenever ``Q / V > 0.10``.
    """

    impact_relative: float
    impact_bps: float
    impact_dollars: float
    sigma_daily: float
    volume_daily: float
    Q: float
    Q_over_V: float
    Y_prefactor: float
    extrapolation_warning: bool


@dataclass(frozen=True)
class CombinedImpactPrediction:
    """Output of Algorithm D — Kyle for small, sqrt for large.

    The crossover ``Q* = Y^2 * sigma^2 / (lambda^2 * V)`` is the order size
    at which the linear-Kyle and sqrt-law impacts agree. Below ``Q*`` the
    linear law is used; above ``Q*`` the sqrt law is used.
    """

    impact_relative: float
    impact_bps: float
    impact_dollars: float
    Q: float
    Q_over_V: float
    Q_crossover: float
    regime: Literal["linear", "sqrt"]
    lambda_hat: float
    Y_prefactor: float
    sigma_daily: float
    volume_daily: float
    extrapolation_warning: bool


@dataclass(frozen=True)
class KyleInputs:
    """Everything ``calibrate``, ``predict``, and ``validate`` consume.

    Produced by ``KyleSqrtImpact.fetch_data``.

    Attributes
    ----------
    ticker:
        Equity / ETF symbol.
    daily_bars:
        Daily OHLCV bars indexed by date. Columns must include ``Open``,
        ``Close``, ``Volume``.
    intraday_bars:
        Optional 5-minute (or other interval) OHLCV bars; ``None`` if the
        intraday lambda is not being computed.
    frequency:
        Which lambda estimator to drive — ``"daily"`` or ``"intraday"``.
    parent_order_shares:
        Default parent-order size used by ``predict``.
    side:
        ``+1`` for buy, ``-1`` for sell.
    Y_prefactor:
        Literature-prior dimensionless sqrt-law prefactor.
    timestamp:
        Assembly timestamp.
    metadata:
        Free-form pass-through for orchestrator hints.
    """

    ticker: str
    daily_bars: pd.DataFrame
    frequency: KyleFrequency
    parent_order_shares: float
    side: int
    Y_prefactor: float
    timestamp: datetime
    intraday_bars: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"KyleInputs.frequency must be one of "
                f"{sorted(VALID_FREQUENCIES)}, got {self.frequency!r}"
            )
        if self.side not in (-1, 1):
            raise ValueError(f"KyleInputs.side must be +/-1, got {self.side}")
        if self.parent_order_shares <= 0:
            raise ValueError(
                f"KyleInputs.parent_order_shares must be > 0, got "
                f"{self.parent_order_shares}"
            )
        if self.Y_prefactor <= 0:
            raise ValueError(
                f"KyleInputs.Y_prefactor must be > 0, got {self.Y_prefactor}"
            )
        for col in ("Open", "Close", "Volume"):
            if col not in self.daily_bars.columns:
                raise ValueError(
                    f"KyleInputs.daily_bars missing required column {col!r}"
                )
        if self.frequency == "intraday" and self.intraday_bars is None:
            raise ValueError(
                "KyleInputs.intraday_bars must be supplied when frequency='intraday'"
            )
        if self.intraday_bars is not None:
            for col in ("Open", "Close", "Volume"):
                if col not in self.intraday_bars.columns:
                    raise ValueError(
                        f"KyleInputs.intraday_bars missing required column {col!r}"
                    )
