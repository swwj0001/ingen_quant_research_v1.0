import numpy as np
import pandas as pd
import pytest

from ingen_qr.features.macro import FEATURES as MACRO_FEATURES
from ingen_qr.features.macro import (
    INTERACTIONS,
    add_interactions,
    first_vintage_market,
    macro_snapshot,
)
from ingen_qr.features.panel import monthly_turnover
from ingen_qr.features.preprocessing import preprocess, preprocess_panel


def market_fixture() -> pd.DataFrame:
    rows = [
        {
            "series_id": series,
            "date": "2010-01-08",
            "value": value,
            "available_at": "2012-01-01T05:00:00Z",
        }
        for series, value in [
            ("DGS3MO", 1),
            ("DGS10", 4),
            ("DBAA", 7),
            ("VIXCLS", 20),
            ("T10Y3M", 999),
            ("BAA10Y", 999),
        ]
    ]
    rows.append(
        {
            "series_id": "DBAA",
            "date": "2010-01-08",
            "value": 9,
            "available_at": "2013-01-01T05:00:00Z",
        }
    )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("risk_free_3m", 0.01),
        ("treasury_10y", 0.04),
        ("baa_yield", 0.07),
        ("vix_level", 20),
        ("term_spread", 0.03),
        ("credit_spread", 0.03),
    ],
)
def test_macro_features_by_hand(name: str, expected: float) -> None:
    market, _ = first_vintage_market(market_fixture())
    values, ages = macro_snapshot(market, pd.Timestamp("2010-01-11T15:00:00Z"))
    assert values[name] == pytest.approx(expected)
    assert ages[name + "_age_days"] == 3


def test_first_vintage_rule_preserves_raw_alfred_data_and_revision_evidence() -> None:
    raw = market_fixture()
    original = raw.copy(deep=True)
    market, revisions = first_vintage_market(raw)
    assert market["available_at"].eq(pd.Timestamp("2010-01-11T14:30:00Z")).all()
    assert market["alfred_available_at"].eq(pd.Timestamp("2012-01-01T05:00:00Z")).all()
    assert revisions.set_index("series_id").loc["DBAA", "revision_share"] == 1
    assert set(market["series_id"]) == {"DGS3MO", "DGS10", "DBAA", "VIXCLS"}
    assert np.isnan(
        macro_snapshot(market, pd.Timestamp("2010-01-11T14:29:00Z"))[0]["baa_yield"]
    )
    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize("name,pair", INTERACTIONS.items())
def test_each_interaction_by_hand(name: str, pair: tuple[str, str]) -> None:
    firm_names = sorted({firm for firm, _ in INTERACTIONS.values()})
    dates = pd.to_datetime(["2020-01-02"] * 3 + ["2020-02-03"] * 3)
    panel = pd.DataFrame({"date": dates, "ticker": ["A", "B", "C"] * 2})
    for firm in firm_names:
        panel[firm] = [10.0, 20.0, 30.0] * 2
    conditions = pd.DataFrame({"date": dates.unique()})
    for macro in MACRO_FEATURES:
        conditions[macro] = [2.0, -6.0]
    result, quality = preprocess_panel(panel, firm_names, conditions)
    firm, macro = pair
    expected = result[firm].to_numpy() * np.array([2.0] * 3 + [-6.0] * 3)
    np.testing.assert_allclose(result[name], expected)
    assert result[name].tolist() == [-2.0, 0.0, 2.0, 6.0, 0.0, -6.0]
    assert not set(MACRO_FEATURES).intersection(result.columns)
    assert set(quality["feature"]) == set(firm_names) | set(INTERACTIONS)


def test_interaction_inputs_must_be_normalized_and_conditions_must_be_unique() -> None:
    firms = sorted({firm for firm, _ in INTERACTIONS.values()})
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "ticker": ["A"],
            **{name: [2.0] for name in firms},
        }
    )
    conditions = pd.DataFrame(
        {"date": frame["date"], **{name: [3.0] for name in MACRO_FEATURES}}
    )
    with pytest.raises(ValueError, match="normalized firm"):
        add_interactions(frame, conditions)
    frame[firms] = 0.0
    with pytest.raises(ValueError, match="one row per date"):
        add_interactions(frame, pd.concat([conditions, conditions]))


@pytest.mark.parametrize("name", [*MACRO_FEATURES, *INTERACTIONS])
def test_macro_levels_and_products_cannot_enter_rank_normalization(name: str) -> None:
    panel = pd.DataFrame(
        {"date": [pd.Timestamp("2020-01-02")], "ticker": ["A"], name: [20.0]}
    )
    with pytest.raises(ValueError, match="Only firm features"):
        preprocess(panel, [name])


def test_missing_input_coverage_is_recorded_before_firm_imputation() -> None:
    names = sorted({firm for firm, _ in INTERACTIONS.values()})
    dates = pd.to_datetime(["2020-01-02"] * 3 + ["2020-02-03"] * 3)
    panel = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"] * 2,
            **{name: [10.0, np.nan, 30.0] * 2 for name in names},
        }
    )
    conditions = pd.DataFrame(
        {"date": dates.unique(), **{name: [20.0, np.nan] for name in MACRO_FEATURES}}
    )
    result, quality = preprocess_panel(panel, names, conditions)
    for name in INTERACTIONS:
        assert result[name].iloc[:3].notna().all()
        assert result[name].iloc[3:].isna().all()
        coverage = quality.loc[quality["feature"] == name, "coverage"].tolist()
        assert coverage == pytest.approx([2 / 3, 0.0])


def test_later_macro_observations_do_not_change_prior_interactions() -> None:
    firms = sorted({firm for firm, _ in INTERACTIONS.values()})
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-02-03"]),
            "ticker": ["A", "A"],
            **{name: [0.5, 0.5] for name in firms},
        }
    )
    conditions = pd.DataFrame(
        {"date": frame["date"], **{name: [2.0, 3.0] for name in MACRO_FEATURES}}
    )
    before = add_interactions(frame, conditions)
    changed = conditions.copy()
    changed.loc[1, MACRO_FEATURES] = 999.0
    after = add_interactions(frame, changed)
    pd.testing.assert_series_equal(before.iloc[0], after.iloc[0])


def test_same_date_preprocessing_is_independent_of_future_and_flags_are_report_only() -> (
    None
):
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 3),
            "ticker": ["A", "B", "C"],
            "x": [1, np.nan, 9],
            "unreported_x": [False, True, False],
        }
    )
    first, quality = preprocess(panel, ["x"])
    assert first["x"].tolist() == [-1, 0, 1]
    assert first["unreported_x"].tolist() == [0, 1, 0]
    future = panel.copy()
    future["date"], future["x"] = pd.Timestamp("2021-01-01"), 12345
    result, _ = preprocess(pd.concat([panel, future], ignore_index=True), ["x"])
    pd.testing.assert_frame_equal(first, result.iloc[:3].reset_index(drop=True))
    assert quality["outliers"].item() == 2


def test_wholly_missing_features_are_neutral_and_reported() -> None:
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")] * 2,
            "ticker": ["A", "B"],
            "x": [np.nan, np.nan],
        }
    )
    result, quality = preprocess(panel, ["x"])
    assert result["x"].eq(0).all()
    assert "unreported_x" not in result
    assert quality["all_missing"].item()


@pytest.mark.parametrize("width,required", [(21, 17), (60, 48)])
def test_turnover_by_hand_zero_volume_and_80_percent_threshold(
    width: int, required: int
) -> None:
    dates = pd.bdate_range("2020-01-01", periods=60)
    stock = pd.DataFrame(
        {
            "date": dates,
            "available_at": pd.to_datetime(dates, utc=True),
            "volume": 100.0,
            "split_ratio": 1.0,
        }
    )
    cutoff = pd.Timestamp(dates[-1]).tz_localize("UTC") + pd.Timedelta(hours=23)
    name = f"turnover_{width}d"
    assert monthly_turnover(stock, dates[-1], cutoff, 10000, dates)[
        name
    ] == pytest.approx(0.01)
    stock.loc[stock.index[-width:], "volume"] = np.nan
    stock.loc[stock.index[-required:], "volume"] = 100
    assert monthly_turnover(stock, dates[-1], cutoff, 10000, dates)[
        name
    ] == pytest.approx(0.01)
    stock.loc[stock.index[-1], "volume"] = np.nan
    assert np.isnan(monthly_turnover(stock, dates[-1], cutoff, 10000, dates)[name])
    stock["volume"] = 0
    assert monthly_turnover(stock, dates[-1], cutoff, 10000, dates)[name] == 0
