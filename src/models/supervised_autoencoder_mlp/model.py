"""`SupervisedAEMLP` — Layer 4 signals & ML model (Jane Street 2021 winner).

Thin orchestration shell around the pure helpers in `signal.py`, the optional
torch training routine in `network.py`, and the entry point in `calibration.py`.

The class:

- Pulls per-ticker price bars via `DataProvider` (one yfinance call per
  ticker, cached).
- Builds the per-(ticker, date) feature panel and the spec's three binary
  targets from forward returns.
- Pools tickers into a single training panel for the ensemble.
- Returns a `Signal` for the latest available date / ticker pair at predict
  time, with the spec's action rule `take_trade = max_k blended_k > tau`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Signal, SignalDirection
from src.models.supervised_autoencoder_mlp.calibration import MODEL_NAME
from src.models.supervised_autoencoder_mlp.calibration import calibrate as _calibrate_impl
from src.models.supervised_autoencoder_mlp.signal import (
    action_rule,
    binarize_targets,
    blend_predictions,
    build_feature_panel,
    impute_with_median,
    median_mad_standardize,
    reconstruction_error,
    rolling_log_returns,
)
from src.models.supervised_autoencoder_mlp.types import (
    DEFAULT_TARGETS,
    EnsembleFit,
    SAEConfig,
    SAEFit,
    SAEInputs,
    TrainConfig,
)

_DEFAULT_PERIOD: str = "2y"
_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10)
_DEFAULT_HORIZON_LABEL: str = "1d"


@register_model
class SupervisedAEMLP(BaseModel):
    """Supervised denoising autoencoder + MLP classifier on tabular features.

    Parameters
    ----------
    universe:
        Tickers contributing rows to the pooled training panel.
    period:
        yfinance ``period`` string for `fetch_prices`. Defaults to ``"2y"``.
    horizons:
        Forward-return horizons (trading days) used to construct the three
        binary targets per spec step 2.
    config:
        `SAEConfig` overrides. ``n_features`` is filled in automatically.
    train_cfg:
        `TrainConfig` overrides (epochs, batch size, seeds, K-fold count).
    device:
        Torch device string for training; ``"cpu"`` is the default.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "weekly"

    def __init__(
        self,
        universe: Sequence[str],
        *,
        period: str = _DEFAULT_PERIOD,
        horizons: tuple[int, ...] = _DEFAULT_HORIZONS,
        config: SAEConfig | None = None,
        train_cfg: TrainConfig | None = None,
        device: str = "cpu",
    ) -> None:
        if len(universe) < 1:
            raise ValueError(
                f"SupervisedAEMLP requires at least 1 ticker, got {len(universe)}"
            )
        if len(horizons) < 1:
            raise ValueError("horizons must contain at least one positive int")
        if any(h < 1 for h in horizons):
            raise ValueError(f"all horizons must be >= 1, got {horizons}")
        self.universe: tuple[str, ...] = tuple(universe)
        self.period: str = period
        self.horizons: tuple[int, ...] = horizons
        self.config_override: SAEConfig | None = config
        self.train_cfg: TrainConfig = train_cfg or TrainConfig()
        self.device: str = device
        self._fit: EnsembleFit | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> SAEInputs:
        feature_frames: list[pd.DataFrame] = []
        target_frames: list[pd.DataFrame] = []
        ticker_id_chunks: list[np.ndarray] = []

        for tid, ticker in enumerate(self.universe):
            bars = provider.fetch_prices(ticker, self.period, _DEFAULT_INTERVAL)
            indexed = _index_bars(bars)
            if indexed is None or indexed.empty:
                continue
            features = build_feature_panel(indexed)
            targets = _forward_return_targets(indexed["Close"], self.horizons)
            # Align features (info at end of day t) with targets (forward
            # returns starting at t+1). Both indices are already date-aligned.
            joined = features.join(targets, how="inner")
            joined = joined.dropna(how="all")
            if joined.empty:
                continue
            f = joined.loc[:, features.columns]
            t = joined.loc[:, targets.columns]
            feature_frames.append(f.assign(_ticker=ticker))
            target_frames.append(t.assign(_ticker=ticker))
            ticker_id_chunks.append(np.full(len(f), tid, dtype=int))

        if not feature_frames:
            raise RuntimeError(
                f"fetch_data produced no usable rows for universe {self.universe!r}"
            )

        feature_panel = pd.concat(feature_frames, axis=0, ignore_index=False)
        target_panel = pd.concat(target_frames, axis=0, ignore_index=False)
        ticker_ids = np.concatenate(ticker_id_chunks)
        dates = feature_panel.index.to_numpy()

        feature_columns = [c for c in feature_panel.columns if c != "_ticker"]
        feature_only = feature_panel.loc[:, feature_columns]
        target_only = target_panel.drop(columns="_ticker")
        # Rebuild targets via the spec's binarization rule.
        binary = binarize_targets(target_only, horizons=self.horizons)
        binary.columns = list(DEFAULT_TARGETS)
        # Drop rows where every feature or every target is NaN — they are
        # unrecoverable. Per-cell NaNs in features are imputed inside calibrate.
        valid = (~feature_only.isna().all(axis=1)) & (~binary.isna().all(axis=1))
        feature_only = feature_only.loc[valid].reset_index(drop=True)
        binary = binary.loc[valid].reset_index(drop=True)
        dates = dates[valid.to_numpy()]
        ticker_ids = ticker_ids[valid.to_numpy()]

        return SAEInputs(
            features=feature_only,
            targets=binary,
            sample_weights=None,
            dates=dates,
            ticker_ids=ticker_ids,
            feature_names=tuple(str(c) for c in feature_only.columns),
            target_names=DEFAULT_TARGETS,
            timestamp=datetime.now(UTC),
            metadata={
                "period": self.period,
                "horizons": list(self.horizons),
                "n_tickers": float(len(self.universe)),
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, SAEInputs):
            raise TypeError(
                f"SupervisedAEMLP.calibrate expects SAEInputs, got "
                f"{type(data).__name__}"
            )
        result = _calibrate_impl(
            data,
            config=self.config_override,
            train_cfg=self.train_cfg,
            device=self.device,
        )
        self._fit = cast(EnsembleFit, result.parameters["ensemble_fit"])
        return result

    def predict(self, data: Any) -> Signal:
        if not isinstance(data, SAEInputs):
            raise TypeError(
                f"SupervisedAEMLP.predict expects SAEInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        # Take the *most recent* row of features for the requested universe.
        latest_idx = int(np.argmax(data.dates))
        x_latest = data.features.iloc[latest_idx : latest_idx + 1].to_numpy(dtype=float)
        ticker_id = int(data.ticker_ids[latest_idx])
        ticker = self.universe[ticker_id] if ticker_id < len(self.universe) else "POOLED"

        blended = self._blend_one(x_latest, fit)  # (1, K)
        actions = action_rule(blended, threshold=self.train_cfg.decision_threshold)
        max_prob = float(blended.max(axis=1)[0])
        direction: SignalDirection = "long" if actions[0] == 1 else "flat"
        ts = pd.Timestamp(data.dates[latest_idx]).to_pydatetime()
        # Ensure tz-aware timestamp for the Signal dataclass.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return Signal(
            ticker=ticker,
            direction=direction,
            strength=float(max(min(max_prob, 1.0), 0.0)),
            timestamp=ts,
            horizon=_DEFAULT_HORIZON_LABEL,
            metadata={
                "probabilities": blended[0].tolist(),
                "n_ensemble_members": float(fit.n_members),
                "decision_threshold": float(self.train_cfg.decision_threshold),
                "target_names": list(fit.target_names),
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, SAEInputs):
            raise TypeError(
                f"SupervisedAEMLP.validate expects SAEInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        X = data.features.to_numpy(dtype=float)
        Y = data.targets.to_numpy(dtype=float)
        valid_mask = np.all(np.isfinite(Y), axis=1)
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        if X_valid.size == 0:
            return {"n_rows": 0, "n_members": fit.n_members}

        from src.models.supervised_autoencoder_mlp.signal import brier_score, roc_auc

        probs, x_hat_standardized = self._predict_all(X_valid, fit)
        # The reconstruction is in standardized space — recover the
        # standardized inputs the network actually saw so the MSE is comparable.
        x_standardized = self._standardize_with_first_member(X_valid, fit)
        recon = reconstruction_error(x_standardized, x_hat_standardized)

        brier = brier_score(probs, Y_valid)
        aucs = [roc_auc(probs[:, k], Y_valid[:, k]) for k in range(probs.shape[1])]
        # Pairwise seed correlation: variance across members of the *first* row.
        per_member_first_row = np.stack(
            [
                self._predict_one_member(X_valid[:1], member, fit.config)[0][0]
                for member in fit.members
            ],
            axis=0,
        )
        seed_corr = float(np.corrcoef(per_member_first_row).mean()) if fit.n_members > 1 else 1.0

        return {
            "n_rows": int(X_valid.shape[0]),
            "n_members": fit.n_members,
            "recon_error_mean": float(recon.mean()),
            "recon_error_p95": float(np.quantile(recon, 0.95)),
            "brier_per_target": {
                name: float(brier[k]) for k, name in enumerate(fit.target_names)
            },
            "auc_per_target": {
                name: float(aucs[k]) for k, name in enumerate(fit.target_names)
            },
            "mean_ensemble_seed_correlation": seed_corr,
        }

    # ----- Public conveniences -----------------------------------------------

    @property
    def fit(self) -> EnsembleFit:
        return self._require_fit()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Run the ensemble on a raw (N, p) feature matrix; return blended probs."""

        fit = self._require_fit()
        return self._blend_one(x, fit)

    # ----- Internal helpers --------------------------------------------------

    def _require_fit(self) -> EnsembleFit:
        if self._fit is None:
            raise RuntimeError(
                "SupervisedAEMLP has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    def _blend_one(self, x: np.ndarray, fit: EnsembleFit) -> np.ndarray:
        probs, _ = self._predict_all(x, fit)
        return probs

    def _predict_all(
        self, x: np.ndarray, fit: EnsembleFit
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run every ensemble member, median-blend predictions, average x_hat."""

        member_probs: list[np.ndarray] = []
        member_xhat: list[np.ndarray] = []
        for member in fit.members:
            probs, x_hat = self._predict_one_member(x, member, fit.config)
            member_probs.append(probs)
            member_xhat.append(x_hat)
        blended_probs = blend_predictions(member_probs)
        mean_xhat = np.mean(np.stack(member_xhat, axis=0), axis=0)
        return blended_probs, mean_xhat

    def _predict_one_member(
        self, x: np.ndarray, member: SAEFit, config: SAEConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        # Apply the per-member standardization, then run the network.
        x_imp, _, _ = impute_with_median(x, median=member.feature_median)
        x_std, _, _ = median_mad_standardize(
            x_imp,
            median=member.feature_median,
            mad=member.feature_mad / 1.4826,
        )
        # Lazy torch import.
        from src.models.supervised_autoencoder_mlp.network import predict_proba

        return predict_proba(
            state_dict=member.state_dict,
            x=x_std,
            config=config,
            device=self.device,
        )

    def _standardize_with_first_member(
        self, x: np.ndarray, fit: EnsembleFit
    ) -> np.ndarray:
        """Standardize using the first member's stats (representative diagnostic)."""

        member = fit.members[0]
        x_imp, _, _ = impute_with_median(x, median=member.feature_median)
        x_std, _, _ = median_mad_standardize(
            x_imp,
            median=member.feature_median,
            mad=member.feature_mad / 1.4826,
        )
        return x_std


# ---- module-level helpers ---------------------------------------------------


def _index_bars(bars: pd.DataFrame) -> pd.DataFrame | None:
    """Pick a DatetimeIndex from a yfinance frame; return None if unusable."""

    if bars is None or bars.empty:
        return None
    df = bars.copy()
    if "Date" in df.columns:
        df.index = pd.DatetimeIndex(pd.to_datetime(df["Date"]))
        df = df.drop(columns="Date")
    elif "Datetime" in df.columns:
        df.index = pd.DatetimeIndex(pd.to_datetime(df["Datetime"]))
        df = df.drop(columns="Datetime")
    else:
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    # Strip timezone so panel concat does not produce mixed-tz indices.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _forward_return_targets(
    close: pd.Series, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """Build the (T, H) forward log-return panel: row t -> log(C_{t+h}/C_t)."""

    out: dict[str, pd.Series] = {}
    for h in horizons:
        # Forward return = backward return shifted up by `h` periods.
        out[f"fwd_{h}"] = rolling_log_returns(close, h).shift(-h)
    return pd.DataFrame(out)


__all__ = ["SupervisedAEMLP"]
