"""Purged group K-fold cross-validation by date (Lopez de Prado, 2018).

The training set excludes any dates within `purge` days of the validation
block on either side (to block leakage through the forward-return target
horizon `h`), plus an additional `embargo` days after the validation
block (to block leakage through serial dependence in the residual).

We always partition the *unique date axis* — never row positions — so the
splits are agnostic to the per-date panel size.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from src.models.gbdt_transformer_ensembles.types import FoldSpec


def purged_group_kfold(
    dates: pd.Series | pd.DatetimeIndex,
    *,
    n_splits: int,
    purge: int,
    embargo: int,
) -> list[FoldSpec]:
    """Build a list of purged K-fold splits on the *unique sorted* dates.

    Parameters
    ----------
    dates:
        The full date column of the panel. Duplicates allowed; only the
        unique sorted values are partitioned.
    n_splits:
        Number of folds. Must be >= 2.
    purge:
        Number of unique-date positions to remove from training on BOTH
        sides of each validation block.
    embargo:
        Additional unique-date positions to remove from training AFTER
        the validation block.
    """

    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if purge < 0:
        raise ValueError(f"purge must be >= 0, got {purge}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    if isinstance(dates, pd.DatetimeIndex):
        unique = np.array(sorted(dates.unique()))
    else:
        unique = np.array(sorted(pd.Series(dates).unique()))
    n_dates = len(unique)
    if n_splits > n_dates:
        raise ValueError(
            f"n_splits={n_splits} exceeds number of unique dates {n_dates}"
        )

    # Equal-size fold blocks; the last fold absorbs the remainder.
    boundaries = np.linspace(0, n_dates, n_splits + 1).astype(int)
    folds: list[FoldSpec] = []
    for k in range(n_splits):
        start, end = boundaries[k], boundaries[k + 1]
        val_idx = np.arange(start, end, dtype=np.int64)
        if val_idx.size == 0:
            continue
        # Training = everything outside [start - purge, end - 1 + purge + embargo].
        block_lo = max(0, start - purge)
        block_hi = min(n_dates, end + purge + embargo)
        mask = np.ones(n_dates, dtype=bool)
        mask[block_lo:block_hi] = False
        train_idx = np.where(mask)[0]
        folds.append(
            FoldSpec(
                fold_id=k,
                train_dates=train_idx.astype(np.int64),
                val_dates=val_idx.astype(np.int64),
            )
        )
    if not folds:
        raise RuntimeError("purged_group_kfold produced no folds (degenerate input)")
    return folds


def iter_row_masks(
    folds: list[FoldSpec],
    dates: pd.Series,
) -> Iterator[tuple[FoldSpec, np.ndarray, np.ndarray]]:
    """For each fold, return `(fold, train_row_mask, val_row_mask)`.

    The mapping from unique-date positions to row positions is built once
    per call so we can stream folds without rebuilding the dict.
    """

    unique = np.array(sorted(pd.Series(dates).unique()))
    date_to_pos = {d: i for i, d in enumerate(unique)}
    date_positions = pd.Series(dates).map(date_to_pos).to_numpy()
    for fold in folds:
        train_set = set(fold.train_dates.tolist())
        val_set = set(fold.val_dates.tolist())
        train_mask = np.fromiter(
            (p in train_set for p in date_positions),
            count=len(date_positions),
            dtype=bool,
        )
        val_mask = np.fromiter(
            (p in val_set for p in date_positions),
            count=len(date_positions),
            dtype=bool,
        )
        yield fold, train_mask, val_mask


__all__ = ["iter_row_masks", "purged_group_kfold"]
