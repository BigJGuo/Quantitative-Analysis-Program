"""Public interface for the Expected Shortfall (FRTB) risk model.

See `models/layer6_risk/18_expected_shortfall.md` for the full specification.

Public surface:

- ``ExpectedShortfall``           — the `BaseModel` subclass registered as
                                    ``"expected_shortfall"``.
- ``ESInputs`` / ``ESFit`` / ``ESResult`` — dataclasses passed across the
                                    model boundary.
- ``calibrate``                   — `ESInputs -> CalibrationResult` entry.
- Pure-math helpers                — `es_gaussian`, `es_student_t`,
                                    `es_historical`, `es_monte_carlo`, plus
                                    `acerbi_szekely_z1` / `_z2` backtests
                                    and the Student-t MLE / PPF / PDF.
"""

from __future__ import annotations

from src.models.expected_shortfall.calibration import MODEL_NAME, calibrate
from src.models.expected_shortfall.model import ExpectedShortfall
from src.models.expected_shortfall.signal import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    apply_liquidity_scaling,
    es_gaussian,
    es_historical,
    es_monte_carlo,
    es_student_t,
    fit_student_t,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    portfolio_pnl,
    rolling_realized_tail_mean,
    student_t_cdf_raw,
    student_t_pdf_raw,
    student_t_ppf_raw,
    traffic_light_band,
)
from src.models.expected_shortfall.types import (
    FRTB_ALPHA,
    VALID_METHODS,
    ESFit,
    ESInputs,
    ESMethod,
    ESResult,
)

__all__ = [
    "FRTB_ALPHA",
    "MODEL_NAME",
    "VALID_METHODS",
    "ESFit",
    "ESInputs",
    "ESMethod",
    "ESResult",
    "ExpectedShortfall",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "apply_liquidity_scaling",
    "calibrate",
    "es_gaussian",
    "es_historical",
    "es_monte_carlo",
    "es_student_t",
    "fit_student_t",
    "normal_cdf",
    "normal_pdf",
    "normal_ppf",
    "portfolio_pnl",
    "rolling_realized_tail_mean",
    "student_t_cdf_raw",
    "student_t_pdf_raw",
    "student_t_ppf_raw",
    "traffic_light_band",
]
