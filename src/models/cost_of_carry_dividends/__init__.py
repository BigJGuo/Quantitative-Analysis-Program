"""Cost-of-Carry / Discrete-Dividend model — Layer 2 structural.

Public surface:

* `CostOfCarry`           — the BaseModel subclass; orchestration entry point.
* `calibrate`             — fit cost band + project forward dividends.
* `compute_result`        — pure transform on `CostOfCarryInputs`.
* Public dataclasses      — Dividend, DividendStream, RateCurve, BasketHolding,
                            CostParameters, BasisDecomposition, CostOfCarryInputs,
                            CostOfCarryResult, ETFBasketResult.

See `models/layer2_structural/05_cost_of_carry_dividends.md` for the math.
"""

from src.models.cost_of_carry_dividends.calibration import (
    MODEL_NAME,
    calibrate,
    project_forward_dividends,
)
from src.models.cost_of_carry_dividends.model import (
    CostOfCarry,
    FetchedData,
    build_default_expiry,
)
from src.models.cost_of_carry_dividends.signal import (
    accrued_dividends,
    aggregate_constituent_dividends,
    basis,
    classify_basis_action,
    classify_etf_action,
    compute_etf_basket_result,
    compute_result,
    equivalent_continuous_yield,
    etf_premium,
    forward_dividend_strip,
    fv_dividend_stream,
    implied_repo_rate,
    interpolate_rate,
    pv_dividend_stream,
    signal_strength,
    theoretical_futures_continuous,
    theoretical_futures_discrete,
    theoretical_futures_price,
    time_to_maturity,
)
from src.models.cost_of_carry_dividends.types import (
    BasisDecomposition,
    BasketHolding,
    CostOfCarryInputs,
    CostOfCarryResult,
    CostParameters,
    Dividend,
    DividendStream,
    ETFBasketResult,
    IndexConstituent,
    RateCurve,
)

__all__ = [
    "MODEL_NAME",
    "BasisDecomposition",
    "BasketHolding",
    "CostOfCarry",
    "CostOfCarryInputs",
    "CostOfCarryResult",
    "CostParameters",
    "Dividend",
    "DividendStream",
    "ETFBasketResult",
    "FetchedData",
    "IndexConstituent",
    "RateCurve",
    "accrued_dividends",
    "aggregate_constituent_dividends",
    "basis",
    "build_default_expiry",
    "calibrate",
    "classify_basis_action",
    "classify_etf_action",
    "compute_etf_basket_result",
    "compute_result",
    "equivalent_continuous_yield",
    "etf_premium",
    "forward_dividend_strip",
    "fv_dividend_stream",
    "implied_repo_rate",
    "interpolate_rate",
    "project_forward_dividends",
    "pv_dividend_stream",
    "signal_strength",
    "theoretical_futures_continuous",
    "theoretical_futures_discrete",
    "theoretical_futures_price",
    "time_to_maturity",
]
