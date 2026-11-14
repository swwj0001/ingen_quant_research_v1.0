"""Per-date transformations; missing flags only for unreported fundamentals."""

import numpy as np
import pandas as pd

from ingen_qr.features.macro import FEATURES as MACRO_FEATURES
from ingen_qr.features.macro import INTERACTIONS, add_interactions


def preprocess(
    panel: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(features).intersection([*MACRO_FEATURES, *INTERACTIONS]):
        raise ValueError("Only firm features may be cross-sectionally normalized.")
    result = panel[["date", "ticker"]].copy()
    records = []
    for date, group in panel.groupby("date", sort=True):
        for name in features:
            raw = group[name].replace([np.inf, -np.inf], np.nan).astype(float)
            valid = raw.dropna()
            low, high = (
                valid.quantile([0.01, 0.99]).tolist()
                if not valid.empty
                else [np.nan, np.nan]
            )
            clipped = raw.clip(low, high)
            filled = clipped.fillna(float(clipped.median()) if not valid.empty else 0.0)
            ranks = filled.rank(method="average")
            normalized = (
                2.0 * (ranks - 1.0) / (len(group) - 1.0) - 1.0
                if len(group) > 1
                else ranks * 0.0
            )
            if filled.nunique() <= 1:
                normalized[:] = 0.0
            result.loc[group.index, name] = normalized
            flag = "unreported_" + name
            if flag in panel:
                result.loc[group.index, flag] = group[flag].astype(int)
            records.append(
                {
                    "date": date,
                    "feature": name,
                    "rows": len(group),
                    "observed": len(valid),
                    "coverage": len(valid) / len(group),
                    "outliers": int(((raw < low) | (raw > high)).sum()),
                    "all_missing": valid.empty,
                    "transformation": "firm_rank_normalization",
                }
            )
    return result, pd.DataFrame(records)


def preprocess_panel(
    panel: pd.DataFrame, firm_features: list[str], conditions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize firms first; append unranked conditional interactions and coverage."""
    normalized, quality = preprocess(panel, firm_features)
    result = add_interactions(normalized, conditions)
    source = panel[["date", "ticker", *firm_features]].merge(
        conditions[["date", *MACRO_FEATURES]],
        on="date",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    records = []
    for date, group in source.groupby("date", sort=True):
        for name, (firm, macro) in INTERACTIONS.items():
            observed = np.isfinite(group[firm]) & np.isfinite(group[macro])
            count = int(observed.sum())
            records.append(
                {
                    "date": date,
                    "feature": name,
                    "rows": len(group),
                    "observed": count,
                    "coverage": count / len(group),
                    "outliers": 0,
                    "all_missing": count == 0,
                    "transformation": "normalized_firm_times_raw_macro",
                }
            )
    return result, pd.concat([quality, pd.DataFrame(records)], ignore_index=True)
