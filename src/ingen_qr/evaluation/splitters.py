"""Expanding development splits with interval purging and a sealed holdout."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

IndexArray = NDArray[np.int64]
HOLDOUT_START = pd.Timestamp("2024-01-01")


@dataclass(frozen=True)
class Fold:
    test_month: pd.Timestamp
    train: IndexArray
    validation: IndexArray
    test: IndexArray
    refit: IndexArray


def fold_split_sha256(meta: pd.DataFrame, fold: Fold) -> str:
    digest = hashlib.sha256()
    digest.update(f"{fold.test_month:%Y-%m-%d}\n".encode())
    identities = meta[["date", "ticker"]]
    groups = (
        ("train", fold.train),
        ("validation", fold.validation),
        ("refit", fold.refit),
        ("test", fold.test),
    )
    for role, indices in groups:
        keys = identities.iloc[indices].copy()
        keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
        keys = keys.sort_values(["date", "ticker"])
        digest.update(f"{role}:{len(indices)}\n".encode())
        payload = keys.to_csv(index=False, header=False, lineterminator="\n")
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def split_manifest_sha256(folds: pd.DataFrame) -> str:
    manifest = folds[["test_month", "split_sha256"]].copy()
    manifest["test_month"] = pd.to_datetime(manifest["test_month"]).dt.strftime(
        "%Y-%m-%d"
    )
    payload = manifest.sort_values("test_month").to_csv(
        index=False, lineterminator="\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_indices(
    meta: pd.DataFrame,
    boundary: pd.Timestamp,
    cutoff: pd.Timestamp,
    embargo_months: int = 1,
) -> IndexArray:
    dates = pd.to_datetime(meta["date"])
    ends = pd.to_datetime(meta["label_end"])
    ready = pd.to_datetime(meta["label_available_at"], utc=True)
    # A complete calendar month separates label completion from the next block.
    embargo_boundary = boundary - pd.DateOffset(months=embargo_months)
    valid = (dates < boundary) & (ends < embargo_boundary) & (ready <= cutoff)
    return np.asarray(np.flatnonzero(valid.to_numpy()), dtype=np.int64)


class PurgedWalkForward:
    def __init__(
        self,
        first_test: str = "2016-01-01",
        last_test: str = "2023-12-01",
        validation_months: int = 12,
        embargo_months: int = 1,
        unlock_holdout: bool = False,
    ) -> None:
        self.first_test = pd.Timestamp(first_test)
        self.last_test = pd.Timestamp(last_test)
        self.validation_months = validation_months
        self.embargo_months = embargo_months
        self.unlock_holdout = unlock_holdout
        if (
            validation_months < 1
            or embargo_months < 0
            or self.last_test < self.first_test
        ):
            raise ValueError("Invalid walk-forward configuration.")
        if not unlock_holdout and self.last_test >= HOLDOUT_START:
            raise PermissionError("Holdout dates require an explicit unlock flag.")

    def split(self, meta: pd.DataFrame) -> Iterator[Fold]:
        dates = pd.to_datetime(meta["date"])
        ends = pd.to_datetime(meta["label_end"])
        if not self.unlock_holdout and (
            (dates >= HOLDOUT_START).any() or (ends >= HOLDOUT_START).any()
        ):
            raise PermissionError(
                "Input includes sealed holdout dates or label windows."
            )
        for month in pd.date_range(self.first_test, self.last_test, freq="MS"):
            next_month = month + pd.DateOffset(months=1)
            test = np.flatnonzero(
                ((dates >= month) & (dates < next_month)).to_numpy()
            ).astype(np.int64)
            if not len(test):
                continue
            validation_start = month - pd.DateOffset(months=self.validation_months)
            validation_cutoff = validation_start.tz_localize("UTC")
            train = training_indices(
                meta, validation_start, validation_cutoff, self.embargo_months
            )
            cutoff = pd.to_datetime(meta.iloc[test]["forecast_at"], utc=True).min()
            ready = pd.to_datetime(meta["label_available_at"], utc=True)
            validation = np.flatnonzero(
                (
                    (dates >= validation_start)
                    & (dates < month)
                    & (ends < month)
                    & (ready <= cutoff)
                ).to_numpy()
            ).astype(np.int64)
            refit = training_indices(meta, month, cutoff, self.embargo_months)
            yield Fold(month, train, validation, test, refit)
