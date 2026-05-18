"""Public interface for the Value-at-Risk model.

See ``models/layer6_risk/17_var.md`` for the full specification.

Public surface:

- ``VaRModel``                  — the `BaseModel` subclass registered as
                                  ``"var"``.
- ``VaRInputs``, ``VaRFit``, ``VaRBacktestResult``
                                — dataclasses passed across the model boundary.
- ``calibrate``                 — pure-function entry point used by the model
                                  (and by external code that already has a
                                  returns panel in hand).
- Pure-math helpers              — `parametric_var`, `historical_var`,
                                  `monte_carlo_var`, `horizon_scale`,
                                  `kupiec_pof_test`,
                                  `christoffersen_independence_test`,
                                  `basel_traffic_light`,
                                  `rolling_var_backtest`,
                                  plus the distribution primitives
                                  `normal_ppf` / `chi2_cdf`.
"""

from __future__ import annotations

from src.models.var.calibration import MODEL_NAME, calibrate
from src.models.var.model import VaRModel
from src.models.var.signal import (
    align_positions,
    basel_traffic_light,
    chi2_cdf,
    christoffersen_independence_test,
    historical_var,
    horizon_scale,
    kupiec_pof_test,
    monte_carlo_var,
    normal_ppf,
    parametric_var,
    portfolio_pnl_vector,
    rolling_var_backtest,
    simple_returns,
)
from src.models.var.types import (
    PORTFOLIO_TICKER,
    VALID_METHODS,
    VALID_TRAFFIC_LIGHTS,
    TrafficLight,
    VaRBacktestResult,
    VaRFit,
    VaRInputs,
    VaRMethod,
)

__all__ = [
    "MODEL_NAME",
    "PORTFOLIO_TICKER",
    "VALID_METHODS",
    "VALID_TRAFFIC_LIGHTS",
    "TrafficLight",
    "VaRBacktestResult",
    "VaRFit",
    "VaRInputs",
    "VaRMethod",
    "VaRModel",
    "align_positions",
    "basel_traffic_light",
    "calibrate",
    "chi2_cdf",
    "christoffersen_independence_test",
    "historical_var",
    "horizon_scale",
    "kupiec_pof_test",
    "monte_carlo_var",
    "normal_ppf",
    "parametric_var",
    "portfolio_pnl_vector",
    "rolling_var_backtest",
    "simple_returns",
]
