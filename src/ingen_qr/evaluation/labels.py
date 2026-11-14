# mypy: disable-error-code=import-untyped

"""Open-to-close 21-session labels with forecast-known risk-free rates."""

import exchange_calendars as xcals
import numpy as np
import pandas as pd

HOLDOUT_START = pd.Timestamp("2024-01-01")


def make_labels(
    cohorts: pd.DataFrame,
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    end: str = "2023-12-31",
    unlock_holdout: bool = False,
) -> pd.DataFrame:
    limit = pd.Timestamp(end)
    if not unlock_holdout and (
        limit >= HOLDOUT_START or (cohorts["date"] >= HOLDOUT_START).any()
    ):
        raise PermissionError(
            "Holdout is locked; development labels must end before 2024-01-01."
        )
    calendar = xcals.get_calendar(
        "XNYS",
        start=cohorts["date"].min() - pd.Timedelta(days=10),
        end=limit + pd.Timedelta(days=60),
    )
    sessions = pd.DatetimeIndex(calendar.sessions)
    exits = {
        entry: sessions[int(sessions.get_indexer(pd.DatetimeIndex([entry]))[0]) + 20]
        for entry in cohorts["date"].unique()
    }
    endpoints = set(exits) | set(exits.values())
    selected = prices.loc[
        (prices["date"] <= limit) & prices["date"].isin(endpoints)
    ].set_index(["ticker", "date"])
    if not selected.index.is_unique:
        raise ValueError("Labels require one price per ticker/session.")
    rates = macro.loc[macro["series_id"] == "DGS3MO"].sort_values("date")
    records = []
    lookup = selected.to_dict("index")
    cutoffs = cohorts.drop_duplicates("date").set_index("date")["forecast_at"].to_dict()
    terms = {}
    for entry, exit_day in exits.items():
        label_end = calendar.session_close(exit_day)
        known_rates = rates.loc[rates["available_at"] <= cutoffs[entry]]
        rate = float(known_rates.iloc[-1]["value"]) if not known_rates.empty else np.nan
        year_fraction = (label_end - calendar.session_open(entry)).total_seconds() / (
            365.0 * 86400.0
        )
        terms[entry] = (exit_day, label_end, rate * year_fraction)
    for row in cohorts.to_dict("records"):
        entry = pd.Timestamp(row["date"])
        exit_day, label_end, risk_free = terms[entry]
        excess, total = np.nan, np.nan
        status = (
            "label_not_mature_by_development_end"
            if exit_day > limit
            else "missing_endpoint_price"
        )
        entry_key, exit_key = (str(row["ticker"]), entry), (
            str(row["ticker"]),
            exit_day,
        )
        ready = label_end + pd.Timedelta(minutes=30)
        if exit_day <= limit and entry_key in lookup and exit_key in lookup:
            first, last = lookup[entry_key], lookup[exit_key]
            opening, closing = float(first["adjusted_open"]), float(
                last["adjusted_close"]
            )
            ready = max(ready, first["available_at"], last["available_at"])
            if opening > 0 and closing > 0 and np.isfinite(opening + closing):
                total = closing / opening - 1.0
                excess = total - risk_free
                status = (
                    "ok"
                    if np.isfinite(excess)
                    else "missing_forecast_known_risk_free_rate"
                )
        records.append(
            {
                "date": entry,
                "ticker": str(row["ticker"]),
                "forecast_at": row["forecast_at"],
                "label_end": exit_day,
                "label_available_at": ready,
                "risk_free_21d": risk_free,
                "label_total_return": total,
                "label_excess_return": excess,
                "label_status": status,
            }
        )
    result = pd.DataFrame(records)
    # Midpoint percentiles give ties one class and keep the middle third missing.
    result["label_rank"] = result.groupby("date")["label_excess_return"].transform(
        lambda x: (x.rank(method="average") - 0.5) / x.notna().sum()
    )
    result["label_class"] = np.where(
        result["label_rank"] <= 1 / 3,
        0.0,
        np.where(result["label_rank"] >= 2 / 3, 1.0, np.nan),
    )
    return result
