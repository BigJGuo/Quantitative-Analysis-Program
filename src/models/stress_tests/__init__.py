"""Public interface for the stress-tests risk model.

See `models/layer6_risk/19_stress_tests.md` for the full specification.
"""

from __future__ import annotations

from src.models.stress_tests.calibration import MODEL_NAME, calibrate
from src.models.stress_tests.model import (
    DEFAULT_CRISIS_WINDOWS,
    DEFAULT_HYPOTHETICALS,
    DEFAULT_SECTOR_MAP,
    StressTests,
)
from src.models.stress_tests.signal import (
    apply_proxy_fill,
    estimate_betas_ols,
    historical_replay_pnl,
    hypothetical_pnl,
    is_plausible,
    portfolio_factor_sensitivities,
    reverse_stress_shock,
    scenario_coverage_matrix,
    slice_window,
    top_contributors,
    window_endpoint_sensitivity,
    window_factor_change,
)
from src.models.stress_tests.types import (
    DEFAULT_FACTORS,
    VALID_SCENARIO_TYPES,
    CrisisWindow,
    HypotheticalScenario,
    ReverseStressResult,
    ScenarioPnL,
    ScenarioType,
    StressFit,
    StressInputs,
    StressReport,
)

__all__ = [
    "DEFAULT_CRISIS_WINDOWS",
    "DEFAULT_FACTORS",
    "DEFAULT_HYPOTHETICALS",
    "DEFAULT_SECTOR_MAP",
    "MODEL_NAME",
    "VALID_SCENARIO_TYPES",
    "CrisisWindow",
    "HypotheticalScenario",
    "ReverseStressResult",
    "ScenarioPnL",
    "ScenarioType",
    "StressFit",
    "StressInputs",
    "StressReport",
    "StressTests",
    "apply_proxy_fill",
    "calibrate",
    "estimate_betas_ols",
    "historical_replay_pnl",
    "hypothetical_pnl",
    "is_plausible",
    "portfolio_factor_sensitivities",
    "reverse_stress_shock",
    "scenario_coverage_matrix",
    "slice_window",
    "top_contributors",
    "window_endpoint_sensitivity",
    "window_factor_change",
]
