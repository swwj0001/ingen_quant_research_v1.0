from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ingen_qr.week4.common import config
from ingen_qr.week4.thematic import choice, matching_size, remedy_comparison
from ingen_qr.week4.thematic_panel import load_candidates, thematic_cohorts

SETTINGS = config(Path("configs/week4_thematic.yaml"))


@pytest.mark.parametrize(
    "changed,expected",
    [
        ({"paired_months": 0}, "shrinkage"),
        ({"mean_difference": -0.005}, "shrinkage"),
        ({"nw_t": None}, "shrinkage"),
        ({"thin_month_share": 0.25}, "pooled"),
        ({"short_history_share": 0.3}, "pooled"),
        ({"fundamental_coverage": 0.5}, "pooled"),
        ({"fundamental_coverage": None, "nw_t": None}, "pooled"),
        ({}, "shrinkage"),
    ],
)
def test_fixed_diagnostic_rule(changed, expected):
    values = {
        "paired_months": 95,
        "mean_difference": 0.02,
        "nw_t": 2.5,
        "thin_month_share": 0.1,
        "short_history_share": 0.1,
        "fundamental_coverage": 0.8,
    }
    assert choice({**values, **changed}, SETTINGS)[0] == expected


def test_listing_cutoff_and_excluded_candidate():
    candidates = load_candidates(Path(SETTINGS["candidates"]))
    assert candidates["needs_verification"].all()
    assert (
        candidates.set_index("ticker").loc["STM", "security"]
        == "us_listed_foreign_equity"
    )
    subset = candidates.loc[candidates["ticker"].isin(["ARM", "FANUY", "STM"])]
    dates = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-09-01", "2023-10-02", "2023-11-01"]),
            "forecast_at": pd.to_datetime(
                ["2023-08-31 21:00Z", "2023-09-29 21:00Z", "2023-10-31 21:00Z"],
                utc=True,
            ),
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["ARM", "FANUY", "STM"],
            "date": pd.to_datetime(["2023-09-14", "2023-08-14", "2023-08-14"]),
            "available_at": pd.to_datetime(
                ["2023-09-14 20:30Z", "2023-08-14 20:30Z", "2023-08-14 20:30Z"],
                utc=True,
            ),
        }
    )
    cohort, coverage = thematic_cohorts(
        subset, prices, {}, dates, {"ARM": 1973239}, 252
    )
    assert cohort["ticker"].value_counts().to_dict() == {"ARM": 2, "STM": 3}
    assert (
        cohort.loc[cohort["ticker"].eq("ARM"), "date"].tolist()
        == dates["date"].iloc[1:].tolist()
    )
    assert "FANUY" in coverage["ticker"].tolist()
    assert coverage.set_index("ticker").loc["FANUY", "candidate_months"] == 0


def test_size_match_is_reproducible_and_uses_actual_counts():
    sp = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")] * 20,
            "prediction": np.arange(20, dtype=float),
            "label_excess_return": np.cos(np.arange(20)),
        }
    )
    a = matching_size(sp, sp.iloc[:10], 10, 1)
    b = matching_size(sp, sp.iloc[:10], 10, 1)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[0, "thematic_stocks"] == 10
    assert a.loc[0, "subsamples"] == 10


@pytest.mark.parametrize("remedy", ["pooled", "shrinkage"])
def test_remedies_fit_only_mature_labels_and_deduplicate(tmp_path, monkeypatch, remedy):
    from types import SimpleNamespace

    from ingen_qr.week4 import thematic

    test_month = pd.Timestamp("2020-06-01")
    dates = pd.to_datetime(
        ["2020-03-02", "2020-03-02", "2020-05-01", "2020-06-01", "2020-06-01"]
    )
    meta = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "A", "A", "B"],
            "label_excess_return": [0.1, 0.2, 50.0, 100.0, 200.0],
            "label_class": [0, 1, 1, 1, 1],
        }
    )
    meta["forecast_at"] = meta["date"].dt.tz_localize("UTC")
    meta["label_end"] = meta["date"] + pd.Timedelta(days=15)
    meta["label_available_at"] = meta["label_end"].dt.tz_localize("UTC")
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    sp = SimpleNamespace(
        meta=meta.assign(ticker=["A", "C", "A", "A", "B"]),
        matrix=features,
        folds=[SimpleNamespace(test_month=test_month, refit=np.array([0, 1]))],
    )
    ref = meta.iloc[3:].assign(prediction=[0.9, 0.1])
    trained = []

    class Forest:
        def __init__(self, *_):
            pass

        def fit(self, x, y, _):
            trained.extend(zip(x.to_dict("records"), y.tolist(), strict=True))

        def predict(self, x):
            return x["feature"].to_numpy()

    monkeypatch.setattr(thematic, "Regressor", Forest)
    monkeypatch.setattr(thematic.baseline, "annual_parameters", lambda: {2020: {}})
    settings = {
        "minimum_training_rows": 3 if remedy == "pooled" else 2,
        "shrink_prior_rows": 4,
    }
    output, audit = thematic.thematic_prediction(
        remedy, sp, meta, features, ref, settings, tmp_path
    )
    assert output["prediction"].notna().all()
    assert audit.loc[0, "theme_training_rows"] == 2
    assert {y for _, y in trained} == {0.1, 0.2}
    assert len(trained) == (3 if remedy == "pooled" else 2)
    if remedy == "pooled":
        assert [x["is_thematic"] for x, _ in trained] == [0.0, 1.0, 1.0]
    else:
        assert audit.loc[0, "thematic_weight"] == pytest.approx(2 / 6)
        assert output.iloc[0]["prediction"] > output.iloc[1]["prediction"]


def test_remedy_comparison_uses_identical_stocks_and_months():
    dates = pd.date_range("2022-01-01", periods=2, freq="MS")
    transfer = pd.DataFrame(
        {
            "date": dates.repeat(12),
            "ticker": [str(i) for i in range(12)] * 2,
            "label_excess_return": np.tile(np.arange(12), 2),
        }
    )
    transfer = transfer.assign(
        model="transfer", prediction=-transfer["label_excess_return"]
    )
    remedy = transfer.iloc[:10].assign(
        model="pooled", prediction=transfer.iloc[:10]["label_excess_return"]
    )
    result, monthly = remedy_comparison(remedy, transfer)
    assert result["paired_months"] == 1
    assert result["before_mean_rank_ic"] == pytest.approx(-1)
    assert result["after_mean_rank_ic"] == pytest.approx(1)
    assert monthly["rank_ic_change"].tolist() == pytest.approx([2])
