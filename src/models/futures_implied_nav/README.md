# Futures-Implied NAV

Layer 2 structural fair-value model for US-listed international ETFs (EFA,
EEM, EWJ, FXI, VWO, ...). Spec: [`models/layer2_structural/02_futures_implied_nav.md`](../../../models/layer2_structural/02_futures_implied_nav.md).

When an ETF's underlying market is closed during US hours, its published
iNAV is stale. This model projects the basket forward using a constrained
ridge regression on liquid hedges (futures, country ETFs, ADRs) and an FX
adjustment, then inverse-variance-blends with the official NAV and the
ETF's own secondary mid:

```
FV_t = w_ETF * M_t  +  w_NAV * NAV_off  +  w_Hedge * FV_hedge
FV_hedge = NAV_close * (1 + beta @ r_H) * (FX_t / FX_close)
```

## Public surface

Importable from `src.models.futures_implied_nav`:

- **Model class.** `FuturesImpliedNAV(BaseModel)` — orchestration entry point.
  Registered as `"futures_implied_nav"` (layer 2, refit-frequency `daily`).
- **Calibration.** `calibrate(panel: HedgePanel, **kwargs) -> CalibrationResult`
  — constrained ridge regression with K-fold CV.
- **Compute helpers.** `compute_result`, `compute_fair_value`,
  `compute_premium`, `classify_action`, `inverse_variance_weights`,
  `single_factor_fair_value`, `multi_source_fair_value`, `implied_basket_return`,
  `fx_adjustment`, `signal_strength`.
- **Panel builder.** `build_panel_from_daily` — assembles a `HedgePanel` from
  daily OHLC bars (close-to-close approximation; see Limitations).
- **Dataclasses.** `HedgeSpec`, `HedgePanel`, `NAVAnchor`, `BetaVector`,
  `SourceVariances`, `BlendWeights`, `CostParameters`,
  `FairValueDecomposition`, `FuturesImpliedNAVInputs`,
  `FuturesImpliedNAVResult`.

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.futures_implied_nav import (
    CostParameters, FuturesImpliedNAV, HedgeSpec,
)

provider = YFinanceProvider()
model = FuturesImpliedNAV(
    etf_ticker="EWJ",
    hedges=(HedgeSpec(ticker="^N225", kind="futures", currency="JPY"),),
    fx_tickers=("JPYUSD=X",),
    history_days=90,
    cost_params=CostParameters(half_spread_bps=2.0, cost_band_bps=5.0),
)
data = model.fetch_data(provider)
calibration = model.calibrate(data)
signal = model.predict(data)
print(signal.metadata["fair_value"], signal.direction, signal.metadata["premium"])
```

## Refit frequency

- `beta` (regression coefficients): **nightly**, after the foreign market
  closes and before the next US open. Stable to within ±0.05 over a week.
- Source variances / blend weights: **nightly**, from the last 30 days of
  realized residuals.
- Bid/ask half-spread: **weekly**, from realized round-trip costs.
- Intraday hot loop (FV recompute): **1–5 seconds**.

## Known limitations

1. **Close-to-close panel approximation.** With yfinance daily bars,
   `build_panel_from_daily` uses close-of-day-d as the NAV anchor and
   close-of-day-d+1 as the next-day target. The spec ideally uses
   foreign-close-to-US-4PM hedge windows; live deployment needs an intraday
   data feed.
2. **NAV anchor.** yfinance has no issuer-published NAV. The default anchor
   uses the prior-day ETF close as a NAV proxy. Pass `nav_anchor_override`
   to inject a real issuer NAV feed.
3. **MSCI EM futures absent in yfinance.** For EEM/VWO use a country-ETF
   basket (EWZ, FXI, EWY, EWT, INDA, ...) as the hedge set instead.
4. **FX-hedged ETFs.** Pass an empty `fx_tickers` tuple and supply
   `nav_anchor_override` with `fx_close = {}` so the FX leg collapses to 1.0.
5. **Free-tier yfinance lag.** Live deployment requires a paid feed; the
   model is fine for backtesting and post-trade analysis.
6. **Holiday mismatches.** When the foreign market is closed (Golden Week,
   Lunar New Year), the residual variance grows; expect `w_NAV` to fall
   and `w_hedge` to rise until the foreign market reopens.

## Related models

- **Model 01 (iNAV / equity ETF):** Model 02 supplies the projected basket
  value when constituents are stale.
- **Model 04 (Factor / PCA):** can reduce a wide hedge set to its principal
  components before regression for better conditioning.
- **Model 05 (Cost-of-carry):** roll-adjusts futures prices across the
  contract month before computing hedge returns.
