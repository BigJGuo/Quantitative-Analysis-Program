"""Dataclasses specific to the HAR-RV model.

Shared `Forecast` / `CalibrationResult` live in `src.core.types`; this module
holds the model-internal shapes that flow between `fetch_data`, `calibrate`,
and `predict`.

The `RVComponents` carrier bundles the four observable variance series that
the HAR family is built from (daily RV, bipower variation, downside/upside
semivariances). `HARFit` is the OLS regression output, including the
Newey-West HAC covariance for inference. `HARRVInputs` is the
`fetch_data -> calibrate / predict` payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

HARSpec = Literal["log", "level"]
RVSource = Literal["intraday", "garman_klass", "yang_zhang", "external"]

VALID_SPECS: frozenset[str] = frozenset({"log", "level"})
VALID_SOURCES: frozenset[str] = frozenset(
    {"intraday", "garman_klass", "yang_zhang", "external"}
)

# Coefficient labels for the design matrix columns: [intercept, daily, weekly, monthly].
HAR_COEF_NAMES: tuple[str, ...] = ("c", "beta_d", "beta_w", "beta_m")


@dataclass(frozen=True)
class RVComponents:
    """Daily realized-variance series and optional decompositions.

    All series share the same `DatetimeIndex` of trading dates. `rv_d` is the
    only required field; the others are populated when the underlying data
    permits (intraday returns required for `bipower` and the semivariances).

    Attributes
    ----------
    rv_d:
        Daily realized variance. NaN / zero days should be cleaned before
        regression construction.
    bipower:
        Bipower variation `BV_t = (pi/2) * (M/(M-1)) * sum_{i>=2} |r_i||r_{i-1}|`.
        Estimator of the continuous-variation component (jump-free).
    rv_minus:
        Negative-return realized semivariance.
    rv_plus:
        Positive-return realized semivariance.
    source:
        How the RV series was constructed (intraday sum-of-squares vs OHLC
        range estimator vs externally supplied).
    n_intraday_bars:
        Average number of intraday bars per trading day. `None` when the
        series was not built from intraday data.
    """

    rv_d: pd.Series
    bipower: pd.Series | None = None
    rv_minus: pd.Series | None = None
    rv_plus: pd.Series | None = None
    source: RVSource = "intraday"
    n_intraday_bars: float | None = None

    def __post_init__(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"RVComponents.source must be one of {sorted(VALID_SOURCES)}, "
                f"got {self.source!r}"
            )
        if not isinstance(self.rv_d, pd.Series):
            raise TypeError(
                f"RVComponents.rv_d must be a pd.Series, got {type(self.rv_d).__name__}"
            )
        for name, opt in (
            ("bipower", self.bipower),
            ("rv_minus", self.rv_minus),
            ("rv_plus", self.rv_plus),
        ):
            if opt is not None and not isinstance(opt, pd.Series):
                raise TypeError(
                    f"RVComponents.{name} must be pd.Series or None, "
                    f"got {type(opt).__name__}"
                )


@dataclass(frozen=True)
class HARFit:
    """Fitted HAR-RV regression coefficients and inference machinery.

    Attributes
    ----------
    coefficients:
        Length-4 vector `[c, beta_d, beta_w, beta_m]`. Always in the spec
        space — i.e., these are coefficients on `log RV` when `spec == "log"`
        and on `RV` itself when `spec == "level"`.
    hac_covariance:
        `(4, 4)` Newey-West HAC covariance of `coefficients`. The diagonal
        gives squared standard errors.
    residuals:
        Length-T regression residuals (in spec space).
    residual_variance:
        OLS `sigma_eps^2 = RSS / (T - K)`. Used for the lognormal correction
        when forecasting from a log-spec fit.
    r_squared:
        In-sample coefficient of determination (in spec space).
    n_obs:
        Effective sample size used in the fit (after the 22-day burn-in).
    hac_lag:
        Newey-West truncation lag actually used.
    spec:
        `"log"` (Corsi's preferred form) or `"level"`.
    horizon:
        Forecast horizon `h` baked into the regression. `h = 1` is one-day
        ahead; `h > 1` uses the multi-period averaged dependent variable.
    fit_index:
        DatetimeIndex aligned to `residuals` — the dates on which the
        regression actually has an observation.
    """

    coefficients: np.ndarray
    hac_covariance: np.ndarray
    residuals: np.ndarray
    residual_variance: float
    r_squared: float
    n_obs: int
    hac_lag: int
    spec: HARSpec
    horizon: int
    fit_index: pd.DatetimeIndex
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coefficients.shape != (4,):
            raise ValueError(
                f"HARFit.coefficients must be length-4, got shape "
                f"{self.coefficients.shape}"
            )
        if self.hac_covariance.shape != (4, 4):
            raise ValueError(
                f"HARFit.hac_covariance must be (4, 4), got "
                f"{self.hac_covariance.shape}"
            )
        if self.residuals.shape != (self.n_obs,):
            raise ValueError(
                f"HARFit.residuals length {self.residuals.shape[0]} does not "
                f"match n_obs {self.n_obs}"
            )
        if len(self.fit_index) != self.n_obs:
            raise ValueError(
                f"HARFit.fit_index length {len(self.fit_index)} does not "
                f"match n_obs {self.n_obs}"
            )
        if self.spec not in VALID_SPECS:
            raise ValueError(
                f"HARFit.spec must be one of {sorted(VALID_SPECS)}, "
                f"got {self.spec!r}"
            )
        if self.horizon < 1:
            raise ValueError(f"HARFit.horizon must be >= 1, got {self.horizon}")
        if self.residual_variance < 0:
            raise ValueError(
                f"HARFit.residual_variance must be non-negative, "
                f"got {self.residual_variance}"
            )

    @property
    def standard_errors(self) -> np.ndarray:
        """Length-4 vector of HAC standard errors (sqrt of diagonal)."""
        return np.sqrt(np.maximum(np.diag(self.hac_covariance), 0.0))

    def t_statistics(self) -> np.ndarray:
        se = self.standard_errors
        out = np.where(se > 0, self.coefficients / np.where(se > 0, se, 1.0), np.nan)
        return out


@dataclass(frozen=True)
class HARRVInputs:
    """Everything `calibrate`, `predict`, and `validate` consume.

    Built by `HARRVModel.fetch_data`. `rv_components` holds the daily series;
    the HAR aggregations (`RV_w`, `RV_m`) are recomputed inside the math
    module from `rv_d`, so we do not freeze them into the input payload.
    """

    ticker: str
    rv_components: RVComponents
    spec: HARSpec
    horizon: int
    timestamp: datetime
    annualize: bool = True
    trading_days_per_year: int = 252
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spec not in VALID_SPECS:
            raise ValueError(
                f"HARRVInputs.spec must be one of {sorted(VALID_SPECS)}, "
                f"got {self.spec!r}"
            )
        if self.horizon < 1:
            raise ValueError(
                f"HARRVInputs.horizon must be >= 1, got {self.horizon}"
            )
        if self.trading_days_per_year <= 0:
            raise ValueError(
                f"HARRVInputs.trading_days_per_year must be positive, "
                f"got {self.trading_days_per_year}"
            )
