"""`GBDTTransformerEnsemble` — Layer 4 signals/ML model.

Thin orchestration shell. The class fetches prices for `universe` via the
`DataProvider`, builds a feature panel, calls `calibrate()` to fit the
ensemble, and exposes per-row and cross-sectional predictions.

See `models/layer4_signals_ml/11_gbdt_transformer_ensembles.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Forecast
from src.models.gbdt_transformer_ensembles.calibration import MODEL_NAME
from src.models.gbdt_transformer_ensembles.calibration import calibrate as _calibrate_impl
from src.models.gbdt_transformer_ensembles.features import (
    build_feature_panel,
    fill_nans_for_linear,
    fill_nans_for_trees,
)
from src.models.gbdt_transformer_ensembles.gbdt import (
    GBDTParams,
    predict_gbdt,
)
from src.models.gbdt_transformer_ensembles.sequence_model import (
    LinearSeqParams,
    predict_linear_seq,
)
from src.models.gbdt_transformer_ensembles.signal import (
    cross_sectional_rank_ic,
    population_stability_index,
    summarize_oof_metrics,
    weighted_r2,
)
from src.models.gbdt_transformer_ensembles.stacking import predict_stacker
from src.models.gbdt_transformer_ensembles.types import (
    EnsembleFit,
    EnsemblePrediction,
    FeatureSpec,
    GBDTFit,
    LinearSeqFit,
    PanelFrame,
    coerce_base_learners,
)

_DEFAULT_INTERVAL: str = "1d"


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `bars` with a tz-naive `DatetimeIndex` and OHLCV columns.

    yfinance frames arrive in three shapes depending on whether `reset_index`
    has been applied: indexed by `Date`, by `Datetime`, or with `Date` as a
    column. We coerce all three to one shape so the feature engineering
    module only has to handle the index-based case.
    """

    df = bars.copy()
    if "Date" in df.columns:
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["Date"])))
        df = df.drop(columns=["Date"])
    elif "Datetime" in df.columns:
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["Datetime"])))
        df = df.drop(columns=["Datetime"])
    else:
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


@register_model
class GBDTTransformerEnsemble(BaseModel):
    """Stacked GBDT + linear sequence model over a daily OHLCV panel.

    Parameters
    ----------
    universe:
        Tickers to include. The benchmark (default ``SPY``) is fetched
        automatically when ``feature_spec.benchmark_ticker`` is set; it
        does not need to appear in `universe`.
    feature_spec:
        Feature-engineering knobs. Defaults match the spec's daily layer.
    base_learners:
        Subset of ``{"gbdt", "linear_seq"}``. Defaults to both.
    gbdt_params, linseq_params:
        Per-base-learner hyperparameters.
    n_splits, purge, embargo:
        Purged K-fold CV hyperparameters on the date axis.
    period:
        yfinance ``period`` string. Default ``"2y"`` matches the spec's
        recommended minimum history.
    stacker_alpha:
        Ridge penalty for the meta-learner.
    non_negative_stacker:
        If True, project the stacker weights to a convex blend.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "weekly"

    def __init__(
        self,
        universe: Sequence[str],
        *,
        feature_spec: FeatureSpec | None = None,
        base_learners: Sequence[str] | None = None,
        gbdt_params: GBDTParams | None = None,
        linseq_params: LinearSeqParams | None = None,
        n_splits: int = 5,
        purge: int = 5,
        embargo: int = 1,
        period: str = "2y",
        stacker_alpha: float = 1.0,
        non_negative_stacker: bool = False,
    ) -> None:
        if len(universe) < 2:
            raise ValueError(
                f"GBDTTransformerEnsemble requires >= 2 tickers, got {len(universe)}"
            )
        self.universe: tuple[str, ...] = tuple(universe)
        self.feature_spec: FeatureSpec = feature_spec or FeatureSpec()
        self.base_learners: tuple[str, ...] = coerce_base_learners(base_learners)
        self.gbdt_params: GBDTParams = gbdt_params or GBDTParams()
        self.linseq_params: LinearSeqParams = linseq_params or LinearSeqParams()
        self.n_splits: int = n_splits
        self.purge: int = purge
        self.embargo: int = embargo
        self.period: str = period
        self.stacker_alpha: float = stacker_alpha
        self.non_negative_stacker: bool = non_negative_stacker
        self._fit: EnsembleFit | None = None
        self._panel: PanelFrame | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> PanelFrame:
        prices_by_ticker: dict[str, pd.DataFrame] = {}
        for ticker in self.universe:
            bars = provider.fetch_prices(ticker, self.period, _DEFAULT_INTERVAL)
            if bars is None or bars.empty:
                continue
            df = _normalize_bars(bars)
            prices_by_ticker[ticker] = df

        if not prices_by_ticker:
            raise RuntimeError(
                f"No prices returned for universe {list(self.universe)}"
            )

        benchmark_close: pd.Series | None = None
        bench_ticker = self.feature_spec.benchmark_ticker
        if bench_ticker:
            bench_bars = provider.fetch_prices(bench_ticker, self.period, _DEFAULT_INTERVAL)
            if bench_bars is not None and not bench_bars.empty:
                bdf = _normalize_bars(bench_bars)
                benchmark_close = bdf["Close"].astype(float)

        panel = build_feature_panel(
            prices_by_ticker,
            spec=self.feature_spec,
            benchmark_close=benchmark_close,
        )
        self._panel = panel
        return panel

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, PanelFrame):
            raise TypeError(
                f"GBDTTransformerEnsemble.calibrate expects PanelFrame, got "
                f"{type(data).__name__}"
            )
        result = _calibrate_impl(
            data,
            feature_spec=self.feature_spec,
            base_learners=self.base_learners,
            n_splits=self.n_splits,
            purge=self.purge,
            embargo=self.embargo,
            gbdt_params=self.gbdt_params,
            linseq_params=self.linseq_params,
            stacker_alpha=self.stacker_alpha,
            non_negative_stacker=self.non_negative_stacker,
        )
        self._fit = cast(EnsembleFit, result.parameters["ensemble_fit"])
        self._panel = data
        return result

    def predict(self, data: Any) -> Forecast:
        """Forecast the forward log return for the last date in the panel.

        Returns a single `Forecast` for the universe's *mean* prediction.
        Use `predict_cross_section()` for per-ticker forecasts.
        """

        if not isinstance(data, PanelFrame):
            raise TypeError(
                f"GBDTTransformerEnsemble.predict expects PanelFrame, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        cs = self.predict_cross_section(data)
        mean_value = float(np.mean(cs.values))
        horizon = f"{fit.target_horizon}d"
        return Forecast(
            ticker="UNIVERSE",
            horizon=horizon,
            value=mean_value,
            timestamp=cs.timestamp,
            metadata={
                "tickers": list(cs.tickers),
                "values": cs.values.tolist(),
                "per_learner": {k: v.tolist() for k, v in cs.per_learner.items()},
            },
        )

    def predict_cross_section(self, data: PanelFrame) -> EnsemblePrediction:
        """Per-ticker forecast from the LAST date in the panel.

        Uses the trained base learners (refit on full data) plus the
        stacker. Returns one prediction per ticker present at the last
        date — matching the spec's cross-sectional inference shape.
        """

        fit = self._require_fit()
        if list(data.feature_names) != list(fit.feature_names):
            raise ValueError(
                "Feature names changed between calibration and prediction"
            )
        last_date = pd.Series(data.meta["date"]).max()
        row_mask = (data.meta["date"] == last_date).to_numpy()
        if row_mask.sum() == 0:
            raise RuntimeError("No rows for the last date — empty cross-section")
        tickers = tuple(
            data.meta.loc[row_mask, "ticker"].astype(str).tolist()
        )

        per_learner: dict[str, np.ndarray] = {}
        for name in fit.base_learner_names:
            base_fit = fit.base_fits[name]
            if isinstance(base_fit, GBDTFit):
                X_tree = fill_nans_for_trees(data.X, self.feature_spec.nan_sentinel)
                pred = predict_gbdt(base_fit, X_tree[row_mask])
            elif isinstance(base_fit, LinearSeqFit):
                X_lin, _ = fill_nans_for_linear(data.X)
                X_lin_df = pd.DataFrame(X_lin, columns=list(data.feature_names))
                pred = predict_linear_seq(
                    base_fit,
                    X_lin_df.loc[row_mask].reset_index(drop=True),
                    data.meta.loc[row_mask, "ticker"].reset_index(drop=True),
                    windows=self.linseq_params.windows,
                    ewma_alpha=self.linseq_params.ewma_alpha,
                )
            else:
                raise TypeError(f"Unsupported base-learner fit: {type(base_fit).__name__}")
            per_learner[name] = pred

        if fit.stacker is not None:
            Z = np.column_stack([per_learner[n] for n in fit.base_learner_names])
            values = predict_stacker(fit.stacker, Z)
        else:
            values = np.mean(
                np.stack([per_learner[n] for n in fit.base_learner_names], axis=0),
                axis=0,
            )

        ts = pd.Timestamp(last_date).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return EnsemblePrediction(
            timestamp=ts,
            tickers=tickers,
            values=values,
            per_learner=per_learner,
            horizon=f"{fit.target_horizon}d",
            metadata={"n_base_learners": len(fit.base_learner_names)},
        )

    def validate(self, data: Any) -> dict[str, Any]:
        """Diagnostics: weighted R^2, rank IC, drift PSI, feature importance."""

        if not isinstance(data, PanelFrame):
            raise TypeError(
                f"GBDTTransformerEnsemble.validate expects PanelFrame, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()

        oof = fit.oof_predictions
        out: dict[str, Any] = dict(summarize_oof_metrics(oof))
        out["cv_metrics"] = dict(fit.cv_metrics)

        # In-sample fit vs OOF — large gap implies overfit.
        in_sample: dict[str, float] = {}
        for name, b_fit in fit.base_fits.items():
            if isinstance(b_fit, GBDTFit):
                X_tree = fill_nans_for_trees(data.X, self.feature_spec.nan_sentinel)
                pred_full = predict_gbdt(b_fit, X_tree)
            elif isinstance(b_fit, LinearSeqFit):
                X_lin, _ = fill_nans_for_linear(data.X)
                X_lin_df = pd.DataFrame(X_lin, columns=list(data.feature_names))
                pred_full = predict_linear_seq(
                    b_fit,
                    X_lin_df,
                    data.meta["ticker"].reset_index(drop=True),
                    windows=self.linseq_params.windows,
                    ewma_alpha=self.linseq_params.ewma_alpha,
                )
            else:
                continue
            in_sample[name] = weighted_r2(
                data.y.to_numpy(), pred_full, data.w.to_numpy()
            )
        out["in_sample_r2"] = in_sample

        # Feature importance for GBDT models.
        importance: dict[str, dict[str, float]] = {}
        for name, b_fit in fit.base_fits.items():
            if not isinstance(b_fit, GBDTFit):
                continue
            gain = b_fit.feature_importance_gain
            if gain.sum() > 0:
                top_idx = np.argsort(gain)[::-1][:10]
                importance[name] = {
                    data.feature_names[i]: float(gain[i]) for i in top_idx
                }
        out["top_features"] = importance

        # OOF cross-sectional rank IC (already inside summarize) plus drift PSI
        # on the residual relative to a 50/50 chronological split.
        if len(oof) >= 100:
            half = len(oof) // 2
            ref = oof["yhat_ensemble"].iloc[:half].to_numpy()
            cur = oof["yhat_ensemble"].iloc[half:].to_numpy()
            out["psi_prediction"] = population_stability_index(ref, cur, n_bins=10)

            # IC of the second-half OOF only — sanity check that the signal
            # didn't decay across the panel.
            second_half = oof.iloc[half:]
            out["second_half_rank_ic"] = cross_sectional_rank_ic(
                second_half["yhat_ensemble"].to_numpy(),
                second_half["y"].to_numpy(),
                second_half["date"],
            )

        out["n_trees_per_learner"] = {
            name: int(b_fit.n_trees)
            for name, b_fit in fit.base_fits.items()
            if isinstance(b_fit, GBDTFit)
        }
        return out

    # ----- Public conveniences -----------------------------------------------

    @property
    def fit(self) -> EnsembleFit:
        return self._require_fit()

    def feature_importance(self) -> pd.DataFrame:
        """GBDT feature importance averaged across all GBDT base learners."""

        fit = self._require_fit()
        names = fit.feature_names
        gain = np.zeros(len(names), dtype=float)
        split = np.zeros(len(names), dtype=float)
        n_gbdt = 0
        for b_fit in fit.base_fits.values():
            if not isinstance(b_fit, GBDTFit):
                continue
            gain = gain + b_fit.feature_importance_gain
            split = split + b_fit.feature_importance_split
            n_gbdt += 1
        if n_gbdt == 0:
            return pd.DataFrame(columns=["feature", "gain", "split"])
        return (
            pd.DataFrame({"feature": list(names), "gain": gain, "split": split})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )

    # ----- Internal helpers --------------------------------------------------

    def _require_fit(self) -> EnsembleFit:
        if self._fit is None:
            raise RuntimeError(
                "GBDTTransformerEnsemble has not been calibrated. Call calibrate(data) first."
            )
        return self._fit


__all__ = ["GBDTTransformerEnsemble"]
