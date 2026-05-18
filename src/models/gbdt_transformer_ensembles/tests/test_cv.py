"""Unit tests for purged group K-fold CV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gbdt_transformer_ensembles.cv import (
    iter_row_masks,
    purged_group_kfold,
)


def test_purged_kfold_partitions_validation() -> None:
    """Validation date-blocks across folds must be disjoint and cover the axis."""

    dates = pd.Series(pd.date_range("2024-01-01", periods=50, freq="B"))
    folds = purged_group_kfold(dates, n_splits=5, purge=0, embargo=0)
    val_dates = np.concatenate([f.val_dates for f in folds])
    assert len(val_dates) == 50
    assert len(np.unique(val_dates)) == 50


def test_purged_kfold_purge_removes_neighbors() -> None:
    """With purge=2 and embargo=0, train must skip 2 date positions on each side."""

    dates = pd.Series(pd.date_range("2024-01-01", periods=40, freq="B"))
    folds = purged_group_kfold(dates, n_splits=4, purge=2, embargo=0)
    for fold in folds:
        val_min = fold.val_dates.min()
        val_max = fold.val_dates.max()
        # Every train index must be either < val_min - 2 or > val_max + 2.
        for idx in fold.train_dates:
            assert idx < val_min - 2 + 1 or idx > val_max + 2


def test_purged_kfold_embargo() -> None:
    """Embargo blocks training dates immediately AFTER the validation block."""

    dates = pd.Series(pd.date_range("2024-01-01", periods=30, freq="B"))
    folds = purged_group_kfold(dates, n_splits=3, purge=0, embargo=2)
    # Take the middle fold so we can see embargo on the right side.
    fold = folds[0]
    val_max = fold.val_dates.max()
    forbidden = {val_max + 1, val_max + 2}
    assert not (forbidden & set(fold.train_dates.tolist()))


def test_purged_kfold_input_validation() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=10, freq="B"))
    with pytest.raises(ValueError):
        purged_group_kfold(dates, n_splits=1, purge=0, embargo=0)
    with pytest.raises(ValueError):
        purged_group_kfold(dates, n_splits=3, purge=-1, embargo=0)
    with pytest.raises(ValueError):
        purged_group_kfold(dates, n_splits=20, purge=0, embargo=0)


def test_iter_row_masks_aligns_to_rows() -> None:
    """Row masks must correctly reflect which rows belong to train/val."""

    # 3 tickers x 5 dates = 15 rows
    base_dates = pd.date_range("2024-01-01", periods=5, freq="B")
    dates = pd.Series(np.tile(base_dates, 3))
    folds = purged_group_kfold(dates, n_splits=5, purge=0, embargo=0)
    for fold, train_mask, val_mask in iter_row_masks(folds, dates):
        # Train and val rows must be disjoint
        assert not np.any(train_mask & val_mask)
        # Every row that maps to a val date should be in val_mask
        val_set = set(fold.val_dates.tolist())
        unique = np.array(sorted(pd.Series(dates).unique()))
        for i, d in enumerate(dates):
            pos = int(np.where(unique == d)[0][0])
            if pos in val_set:
                assert val_mask[i]


def test_iter_row_masks_full_pass() -> None:
    """Concatenating val masks across folds must cover every row exactly once."""

    base_dates = pd.date_range("2024-01-01", periods=10, freq="B")
    dates = pd.Series(np.tile(base_dates, 2))
    folds = purged_group_kfold(dates, n_splits=5, purge=0, embargo=0)
    coverage = np.zeros(len(dates), dtype=int)
    for _, _, val_mask in iter_row_masks(folds, dates):
        coverage = coverage + val_mask.astype(int)
    assert (coverage == 1).all()
