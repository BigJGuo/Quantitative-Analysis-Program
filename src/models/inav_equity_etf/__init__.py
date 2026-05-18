"""Public interface for the iNAV / equity-ETF fair-value model.

See `models/layer2_structural/01_inav_equity_etf.md` for the full specification.
"""

from __future__ import annotations

from src.models.inav_equity_etf.calibration import MODEL_NAME, calibrate
from src.models.inav_equity_etf.model import INAVEquityETF
from src.models.inav_equity_etf.signal import (
    accrue_cash,
    accrue_liabilities,
    action_to_signal_direction,
    basket_value_usd,
    classify_action,
    compute_decomposition,
    compute_inav_result,
    compute_premium,
    is_quote_stale,
    signal_strength,
)
from src.models.inav_equity_etf.types import (
    BasketHolding,
    CashLedgerSeed,
    ConstituentQuote,
    CostParameters,
    INAVDecomposition,
    INAVInputs,
    INAVResult,
)

__all__ = [
    "MODEL_NAME",
    "BasketHolding",
    "CashLedgerSeed",
    "ConstituentQuote",
    "CostParameters",
    "INAVDecomposition",
    "INAVEquityETF",
    "INAVInputs",
    "INAVResult",
    "accrue_cash",
    "accrue_liabilities",
    "action_to_signal_direction",
    "basket_value_usd",
    "calibrate",
    "classify_action",
    "compute_decomposition",
    "compute_inav_result",
    "compute_premium",
    "is_quote_stale",
    "signal_strength",
]
