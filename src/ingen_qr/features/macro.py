# mypy: disable-error-code=import-untyped

"""First-vintage market observations with a declared next-session assumption."""

import exchange_calendars as xcals
import numpy as np
import pandas as pd

SERIES = ["DGS3MO", "DGS10", "DBAA", "VIXCLS"]
FEATURES = [
    "risk_free_3m",
    "treasury_10y",
    "baa_yield",
    "vix_level",
    "term_spread",
    "credit_spread",
]
INTERACTIONS = {
    "momentum_x_term": ("momentum_252_21d", "term_spread"),
    "beta_x_credit": ("beta_252d", "credit_spread"),
    "volatility_x_vix": ("volatility_60d", "vix_level"),
    "value_x_term": ("book_to_market", "term_spread"),
    "size_x_vix": ("log_market_cap", "vix_level"),
}


def first_vintage_market(fred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = fred.loc[fred["series_id"].isin(SERIES)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame.loc[np.isfinite(frame["value"])].copy()
    if frame.empty:
        raise ValueError("No first-vintage market observations.")
    counts = frame.groupby(["series_id", "date"])["value"].nunique()
    audit = (
        counts.gt(1)
        .groupby("series_id")
        .agg(["sum", "count", "mean"])
        .rename(
            columns={
                "sum": "dates_with_revisions",
                "count": "dates_observed",
                "mean": "revision_share",
            }
        )
        .reset_index()
    )
    frame = frame.sort_values(["series_id", "date", "available_at"]).drop_duplicates(
        ["series_id", "date"], keep="first"
    )
    frame["alfred_available_at"] = frame["available_at"]
    calendar = xcals.get_calendar(
        "XNYS",
        start=frame["date"].min() - pd.Timedelta(days=5),
        end=frame["date"].max() + pd.Timedelta(days=15),
    )
    next_opens = {
        day: calendar.session_open(
            calendar.date_to_session(day + pd.Timedelta(days=1), direction="next")
        )
        for day in frame["date"].unique()
    }
    frame["available_at"] = pd.to_datetime(frame["date"].map(next_opens), utc=True)
    frame["assumed_availability"] = True
    frame["value"] = np.where(
        frame["series_id"] == "VIXCLS", frame["value"], frame["value"] / 100.0
    )
    frame["first_vintage_not_verified_first_release"] = True
    return frame.reset_index(drop=True), audit


def macro_snapshot(
    market: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[dict[str, float], dict[str, object]]:
    known = market.loc[market["available_at"] <= cutoff].sort_values("date")
    values: dict[str, float] = {}
    metadata: dict[str, object] = {}
    today = cutoff.tz_convert("America/New_York").tz_localize(None).normalize()
    for series, name in zip(SERIES, FEATURES[:4], strict=True):
        rows = known.loc[known["series_id"] == series]
        values[name] = float(rows.iloc[-1]["value"]) if not rows.empty else float("nan")
        metadata[name + "_age_days"] = (
            (today - rows.iloc[-1]["date"]).days if not rows.empty else np.nan
        )
        metadata[name + "_observation_date"] = (
            rows.iloc[-1]["date"] if not rows.empty else pd.NaT
        )
    for name, left, right in [
        ("term_spread", "DGS10", "DGS3MO"),
        ("credit_spread", "DBAA", "DGS10"),
    ]:
        pairs = (
            known.loc[known["series_id"].isin([left, right])]
            .pivot(index="date", columns="series_id", values="value")
            .reindex(columns=[left, right])
            .dropna()
        )
        values[name] = (
            float(pairs.iloc[-1][left] - pairs.iloc[-1][right])
            if not pairs.empty
            else float("nan")
        )
        metadata[name + "_age_days"] = (
            (today - pairs.index[-1]).days if not pairs.empty else np.nan
        )
    return values, metadata


def add_interactions(
    normalized_firms: pd.DataFrame, conditions: pd.DataFrame
) -> pd.DataFrame:
    """Multiply normalized firm ranks by raw macro levels; never rank products."""
    if conditions["date"].duplicated().any():
        raise ValueError("Macro conditions must contain exactly one row per date.")
    for firm in {pair[0] for pair in INTERACTIONS.values()}:
        if normalized_firms[firm].dropna().abs().gt(1 + 1e-12).any():
            raise ValueError("Interaction inputs must be normalized firm features.")
    result = normalized_firms.merge(
        conditions[["date", *FEATURES]],
        on="date",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    for name, (firm, macro) in INTERACTIONS.items():
        result[name] = result[firm] * result[macro]
    return result.drop(columns=FEATURES)
