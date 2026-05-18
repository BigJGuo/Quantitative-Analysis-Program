"""Futures-Implied NAV — Layer 2 structural fair-value model for international ETFs.

Public surface:

* `FuturesImpliedNAV`   — the BaseModel subclass; orchestration entry point.
* `calibrate`            — fit beta from a HedgePanel; returns CalibrationResult.
* `compute_result`       — pure intraday FV computation from inputs.
* `build_panel_from_daily` — convenience builder for the calibration panel.
* Public dataclasses     — HedgeSpec, HedgePanel, NAVAnchor, CostParameters,
                           BetaVector, SourceVariances, BlendWeights,
                           FairValueDecomposition, FuturesImpliedNAVInputs,
                           FuturesImpliedNAVResult.

See `models/layer2_structural/02_futures_implied_nav.md` for the math.
"""

from src.models.futures_implied_nav.calibration import calibrate
from src.models.futures_implied_nav.model import (
    FetchedData,
    FuturesImpliedNAV,
    build_panel_from_daily,
)
from src.models.futures_implied_nav.signal import (
    classify_action,
    compute_fair_value,
    compute_premium,
    compute_result,
    fx_adjustment,
    implied_basket_return,
    inverse_variance_weights,
    multi_source_fair_value,
    signal_strength,
    single_factor_fair_value,
)
from src.models.futures_implied_nav.types import (
    BetaVector,
    BlendWeights,
    CostParameters,
    FairValueDecomposition,
    FuturesImpliedNAVInputs,
    FuturesImpliedNAVResult,
    HedgePanel,
    HedgeSpec,
    NAVAnchor,
    SourceVariances,
)

__all__ = [
    "BetaVector",
    "BlendWeights",
    "CostParameters",
    "FairValueDecomposition",
    "FetchedData",
    "FuturesImpliedNAV",
    "FuturesImpliedNAVInputs",
    "FuturesImpliedNAVResult",
    "HedgePanel",
    "HedgeSpec",
    "NAVAnchor",
    "SourceVariances",
    "build_panel_from_daily",
    "calibrate",
    "classify_action",
    "compute_fair_value",
    "compute_premium",
    "compute_result",
    "fx_adjustment",
    "implied_basket_return",
    "inverse_variance_weights",
    "multi_source_fair_value",
    "signal_strength",
    "single_factor_fair_value",
]
