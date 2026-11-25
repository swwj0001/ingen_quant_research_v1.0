from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ingen_qr import week3
from ingen_qr.evaluation.splitters import Fold, fold_split_sha256, split_manifest_sha256
from ingen_qr.evaluation.week3_samples import Baseline, Week3Sample
from ingen_qr.week3 import score_validation


def test_split_hash_tracks_membership_without_row_order() -> None:
    meta = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-01-02", "2015-01-02", "2016-01-04"]),
            "ticker": ["AAA", "BBB", "CCC"],
        }
    )
    fold = Fold(
        pd.Timestamp("2016-01-01"),
        np.array([0]),
        np.array([1]),
        np.array([2]),
        np.array([0, 1]),
    )
    digest = fold_split_sha256(meta, fold)
    order = np.array([2, 0, 1])
    positions = np.argsort(order)
    reordered = replace(
        fold,
        train=positions[fold.train],
        validation=positions[fold.validation],
        test=positions[fold.test],
        refit=positions[fold.refit],
    )
    assert fold_split_sha256(meta.iloc[order], reordered) == digest
    week2 = pd.DataFrame({"test_month": ["2016-01-01"], "split_sha256": [digest]})
    week3 = week2.assign(split_sha256=fold_split_sha256(meta.iloc[order], reordered))
    assert split_manifest_sha256(week2) == split_manifest_sha256(week3)
    changed = meta.copy()
    changed.loc[2, "ticker"] = "DDD"
    assert fold_split_sha256(changed, fold) != digest


def test_tuning_uses_only_training_and_validation_indices() -> None:
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    meta = X.rename(columns={"x": "label_excess_return"}).assign(
        date=pd.Timestamp("2015-06-01")
    )
    fold = Fold(
        pd.Timestamp("2016-01-01"),
        np.array([0]),
        np.arange(1, 11),
        np.array([11]),
        np.arange(11),
    )
    sample = Week3Sample(meta, X, pd.DataFrame(), [fold], {}, "")

    class SpyModel:
        def fit(self, X, y, meta):
            assert list(X.index) == list(y.index) == list(meta.index) == [0]

        def predict(self, X):
            assert list(X.index) == list(range(1, 11))
            return X["x"].to_numpy()

        def get_params(self):
            return {}

    assert np.isclose(score_validation(SpyModel(), sample, fold), 1.0)


def test_smoke_and_formal_trials_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    monkeypatch.chdir(tmp_path)
    database = Path("mlruns/tracking.db")
    database.parent.mkdir()
    database.touch()
    months = pd.to_datetime(
        ["2013-01-02", "2014-01-02", "2015-01-02", "2016-01-04", "2016-02-01"]
    )
    x = np.tile(np.arange(12, dtype=float), len(months))
    meta = pd.DataFrame(
        {
            "date": months.repeat(12),
            "ticker": np.tile([f"T{i:02d}" for i in range(12)], len(months)),
            "label_excess_return": x,
            "label_end": months.repeat(12) + pd.Timedelta(days=21),
            "label_class": x > 6,
        }
    )
    folds = [
        Fold(
            months[i].replace(day=1),
            np.arange(12),
            np.arange(12, 24),
            np.arange(i * 12, (i + 1) * 12),
            np.arange(24),
        )
        for i in (2, 3, 4)
    ]
    sample = Week3Sample(meta, pd.DataFrame({"x": x}), pd.DataFrame(), folds, {}, "abc")

    class SpyModel:
        def __init__(self, name, params, seed):
            self.params = params

        def fit(self, X, y, meta):
            pass

        def predict(self, X):
            return X["x"].to_numpy()

        def get_params(self):
            return self.params

    monkeypatch.setattr(week3, "Regressor", SpyModel)
    monkeypatch.setattr(
        week3, "read_baseline", lambda _: Baseline("FINISHED", 0, 1, {}, "abc")
    )
    monkeypatch.setattr(week3, "load_verified_sample", lambda *_: sample)
    config = Path("week3.yaml")
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "trials_per_model": 1,
                "experiment": "ingen_qr_week3",
                "output_directory": "data/week3",
                "expected_model_columns": 1,
                "models": {
                    name: {"fixed": {}, "search": {}}
                    for name in (
                        "ridge",
                        "elastic_net",
                        "random_forest",
                        "lightgbm",
                        "adaboost",
                        "mlp",
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    previous_uri = mlflow.get_tracking_uri()
    try:
        smoke = week3.run(config, smoke=True)
        assert smoke.parent == Path("data/week3_smoke")
        assert mlflow.get_experiment_by_name("ingen_qr_week3") is None
        assert mlflow.get_run(smoke.name).data.params["evaluated_folds"] == "2"
        trials = mlflow.search_runs(
            experiment_names=[week3.SMOKE_EXPERIMENT],
            filter_string="tags.run_kind = 'trial'",
        )
        assert trials.groupby("params.model").size().tolist() == [2] * 6
        assert sorted(trials["params.cumulative_trial"].astype(int)) == list(
            range(1, 13)
        )
        predictions = pd.read_parquet(smoke / "predictions.parquet")
        assert set(predictions["date"]) == set(months[2:4])

        formal = week3.run(config)
        assert formal.parent == Path("data/week3")
        assert mlflow.get_run(formal.name).data.params["cumulative_trial_start"] == "2"
        assert mlflow.get_run(formal.name).data.params["cumulative_trial_end"] == "14"
    finally:
        mlflow.set_tracking_uri(previous_uri)
