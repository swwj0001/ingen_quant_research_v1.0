# mypy: disable-error-code=import-untyped

import exchange_calendars as xcals
import pandas as pd

ANCHORS = {
    "TSLA": "2020-12-21",
    "GEHC": "2023-01-04",
}


def audit_membership(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, expected in ANCHORS.items():
        contains = pd.Series(
            [ticker in value.split(",") for value in history["tickers"]],
            index=history.index,
        )
        observed = history.loc[contains, "date"].min()
        rows.append(
            {
                "ticker": ticker,
                "expected": expected,
                "observed": str(observed.date()),
                "matches": observed == pd.Timestamp(expected),
            }
        )
    result = pd.DataFrame(rows)
    if not result["matches"].all():
        raise ValueError("Historical membership disagrees with the official anchors.")
    return result


def build_cohorts(
    history: pd.DataFrame, start: str, end: str, known_through: str
) -> pd.DataFrame:
    calendar = xcals.get_calendar(
        "XNYS",
        start=history["date"].min() - pd.Timedelta(days=14),
        end=pd.Timestamp(end) + pd.Timedelta(days=14),
    )
    reference = history.sort_values("date").copy()
    reference["available_at"] = reference["date"].map(
        lambda day: calendar.session_open(
            calendar.date_to_session(day, direction="next")
        )
    )
    sessions = calendar.sessions_in_range(start, end)
    entries = sessions.to_series().groupby(sessions.to_period("M")).first()
    rows = []
    for entry in entries:
        previous = calendar.previous_session(entry)
        forecast_at = calendar.session_close(previous) + pd.Timedelta(hours=1)
        if (
            forecast_at.tz_convert("America/New_York").date()
            > pd.Timestamp(known_through).date()
        ):
            raise ValueError("A forecast needs membership beyond the pinned source.")
        eligible = reference.loc[reference["available_at"] <= forecast_at]
        latest = eligible.iloc[-1]
        for ticker in latest["tickers"].split(","):
            rows.append(
                {
                    "date": entry,
                    "ticker": ticker,
                    "forecast_at": forecast_at,
                    "membership_effective_date": latest["date"],
                    "available_at": latest["available_at"],
                    "availability_rule": "effective_session_open_proxy",
                }
            )
    return pd.DataFrame(rows)
