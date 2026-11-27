"""As-of firm-month joins and cross-fitted supervised embedding scores."""

from collections import defaultdict
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ingen_qr.evaluation.splitters import training_indices
from ingen_qr.week4.common import KEYS

FILING_GROUPS = {"periodic": ("10-K", "10-Q"), "8k": ("8-K",)}
LEVELS = ["lm_tone", "lm_uncertainty", "lm_litigious", "finbert_tone"]
TONE_CHANGES = ["lm_tone_change", "finbert_tone_change"]
CHANGES = [*TONE_CHANGES, "embedding_change"]
COUNT_FEATURES = ["new_8k_count", "new_10q_count", "new_10k_count"]
EVENT_FEATURES = [*(f"new_filing_{group}" for group in FILING_GROUPS), *COUNT_FEATURES]
EMBEDDING_LEVELS = [f"embedding_score_{group}" for group in FILING_GROUPS]
SENTIMENT_CHANGES = [f"{name}_{group}" for group in FILING_GROUPS for name in CHANGES]
FEATURES = [
    *(
        f"{name}_{group}"
        for group in FILING_GROUPS
        for name in [
            *LEVELS,
            *CHANGES,
            "embedding_score",
            "days_since_filing",
            "new_filing",
        ]
    ),
    "annual_text_similarity",
    *COUNT_FEATURES,
]
FILING_TIMESTAMPS = [
    *(
        f"{name}_{group}"
        for group in FILING_GROUPS
        for name in ("available_at", "previous_available_at")
    ),
    "annual_similarity_available_at",
]


def comparables(documents: pd.DataFrame) -> pd.DataFrame:
    regular: dict[tuple[Any, ...], list[dict[Any, Any]]] = defaultdict(list)
    result = []
    for row in documents.sort_values(
        ["available_at", "accepted_utc", "accession"]
    ).to_dict("records"):
        form = row["form"]
        key = (row["cik"], form, row["items"] if form == "8-K" else "")
        period = pd.to_datetime(row["report_date"], errors="coerce")
        prior = None
        for candidate in reversed(regular[key]):
            if form == "8-K":
                # Match item signature and separate events. No item metadata means no comparable.
                match = (
                    bool(row["items"])
                    and pd.notna(period)
                    and candidate["period"] < period
                )
            elif pd.notna(period) and pd.notna(candidate["period"]):
                gap = (period - candidate["period"]).days
                match = 300 <= gap <= 430 if form == "10-K" else 50 <= gap <= 225
            else:
                match = False
            if match:
                prior = candidate
                break
        regular[key].append({**row, "period": period})
        result.append(
            {
                **row,
                "previous_accession": None if prior is None else prior["accession"],
                "previous_available_at": (
                    pd.NaT if prior is None else prior["available_at"]
                ),
                "lm_tone_change": (
                    np.nan if prior is None else row["lm_tone"] - prior["lm_tone"]
                ),
                "finbert_tone_change": (
                    np.nan
                    if prior is None
                    else row["finbert_tone"] - prior["finbert_tone"]
                ),
                "annual_text_similarity": (
                    float(np.dot(row["embedding"], prior["embedding"]))
                    if form == "10-K"
                    and prior is not None
                    and row["embedding"] is not None
                    and prior["embedding"] is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(result)


def asof_features(
    rows: pd.DataFrame, documents: pd.DataFrame, max_age: int
) -> pd.DataFrame:
    snapshots = rows[[*KEYS, "cik", "forecast_at"]].copy()
    snapshots["previous_cutoff"] = snapshots["date"].dt.to_period("M")
    # Use the actual prior monthly forecast cutoff, even when an issuer skips a month.
    cutoffs = snapshots.drop_duplicates("previous_cutoff").set_index("previous_cutoff")[
        "forecast_at"
    ]
    snapshots["previous_cutoff"] = (snapshots["previous_cutoff"] - 1).map(cutoffs)
    snapshots["previous_cutoff"] = snapshots["previous_cutoff"].fillna(
        snapshots["forecast_at"] - pd.Timedelta(days=31)
    )
    columns = [
        "accession",
        "form",
        "available_at",
        "previous_accession",
        "previous_available_at",
        *LEVELS,
        *TONE_CHANGES,
    ]
    if snapshots["cik"].isna().any():
        raise ValueError(
            "The Week 3 firm-month matrix must resolve to a historical CIK."
        )
    snapshots["cik"] = snapshots["cik"].astype("int64")
    snapshots["forecast_at"] = pd.to_datetime(
        snapshots["forecast_at"], utc=True
    ).dt.as_unit("us")
    snapshots = snapshots.sort_values("forecast_at", kind="stable")
    docs = comparables(documents) if not documents.empty else documents.copy()
    docs = docs.reindex(columns=["cik", *columns, "annual_text_similarity"])
    docs["cik"] = docs["cik"].astype("int64")
    docs["available_at"] = pd.to_datetime(docs["available_at"], utc=True).dt.as_unit(
        "us"
    )
    docs["previous_available_at"] = pd.to_datetime(
        docs["previous_available_at"], utc=True
    )

    join_keys = snapshots[["cik", "forecast_at"]]
    joined = snapshots.reset_index(drop=True).copy()
    for family, forms in FILING_GROUPS.items():
        selected = pd.merge_asof(
            join_keys,
            docs.loc[docs["form"].isin(forms), ["cik", *columns]],
            left_on="forecast_at",
            right_on="available_at",
            by="cik",
            tolerance=pd.Timedelta(days=max_age),
        )
        selected["days_since_filing"] = (
            selected["forecast_at"] - selected["available_at"]
        ).dt.total_seconds() / 86400
        selected["new_filing"] = (
            selected["available_at"].gt(joined["previous_cutoff"]).astype(int)
        )
        selected.loc[selected["new_filing"].eq(0), TONE_CHANGES] = np.nan
        joined = pd.concat(
            [
                joined,
                selected.drop(columns=["cik", "forecast_at"]).add_suffix("_" + family),
            ],
            axis=1,
        )

    yearly = docs.loc[
        docs["form"].eq("10-K"), ["cik", "available_at", "annual_text_similarity"]
    ]
    annual = pd.merge_asof(
        join_keys,
        yearly,
        left_on="forecast_at",
        right_on="available_at",
        by="cik",
        tolerance=pd.Timedelta(days=430),
    )
    joined["annual_text_similarity"] = annual["annual_text_similarity"]
    joined["annual_similarity_available_at"] = annual["available_at"]
    all_dates = joined["forecast_at"].astype("int64").to_numpy()
    prior_dates = joined["previous_cutoff"].dt.as_unit("us").astype("int64").to_numpy()
    for form in ["8-K", "10-Q", "10-K"]:
        count = np.zeros(len(joined), dtype=int)
        by_cik = {
            int(str(issuer)): frame["available_at"].astype("int64").to_numpy()
            for issuer, frame in docs.loc[docs["form"].eq(form)].groupby("cik")
        }
        for issuer, positions in joined.groupby("cik").indices.items():
            issuer_key = int(str(issuer))
            if issuer_key in by_cik:
                times = by_cik[issuer_key]
                count[positions] = np.searchsorted(
                    times, all_dates[positions], side="right"
                ) - np.searchsorted(times, prior_dates[positions], side="right")
        joined[f"new_{form.lower().replace('-', '')}_count"] = count
    return (
        joined.drop(columns="previous_cutoff").sort_values(KEYS).reset_index(drop=True)
    )


def supervised(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    documents: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each historical row once, using only matured, purged earlier labels."""
    result = features.copy().reset_index(drop=True)

    columns = [*KEYS, "label_available_at", "label_end", "label_excess_return"]
    data = result.merge(meta[columns], on=KEYS, how="left", validate="one_to_one")
    vectors = (
        {}
        if documents.empty
        else {
            accession: np.asarray(vector, dtype=float)
            for accession, vector in zip(
                documents["accession"], documents["embedding"], strict=True
            )
            if vector is not None and len(vector) > 0
        }
    )
    Ridge = import_module("sklearn.linear_model").Ridge
    minimum = int(settings["minimum_head_training_rows"])
    embargo = int(settings["head_embargo_months"])
    if embargo < 1 or minimum < 2:
        raise ValueError(
            "Method (c) needs at least one embargo month and two training rows."
        )
    candidates = {
        family: data[f"accession_{family}"].isin(vectors).to_numpy()
        for family in FILING_GROUPS
    }
    audit = []
    for family in FILING_GROUPS:
        accession = f"accession_{family}"
        previous_accession = f"previous_accession_{family}"
        score = f"embedding_score_{family}"
        change = f"embedding_change_{family}"
        result[[score, change]] = np.nan
        for date, group in data.groupby("date", sort=True):
            started = perf_counter()
            cutoff = pd.Timestamp(str(group["forecast_at"].min()))
            boundary = pd.Timestamp(str(date)).replace(day=1)
            positions = training_indices(data, boundary, cutoff, embargo)
            train = data.iloc[positions[candidates[family][positions]]].dropna(
                subset=["label_excess_return"]
            )
            eligible = group.loc[group[accession].isin(vectors)]
            status = "too_few_mature_labels"
            if len(train) >= minimum and not eligible.empty:
                head = Ridge(alpha=float(settings["embedding_ridge_alpha"]))
                head.fit(
                    np.stack([vectors[acc] for acc in train[accession]]),
                    train["label_excess_return"],
                )
                result.loc[eligible.index, score] = head.predict(
                    np.stack([vectors[acc] for acc in eligible[accession]])
                )
                compared = eligible.loc[
                    eligible[f"new_filing_{family}"].eq(1)
                    & eligible[previous_accession].isin(vectors)
                ]
                if not compared.empty:
                    previous = head.predict(
                        np.stack([vectors[acc] for acc in compared[previous_accession]])
                    )
                    result.loc[compared.index, change] = (
                        result.loc[compared.index, score].to_numpy() - previous
                    )
                status = "fitted"
            elif eligible.empty:
                status = "no_embeddings_at_forecast"
            audit.append(
                {
                    "filing_group": family,
                    "date": date,
                    "forecast_at": cutoff,
                    "training_rows": len(train),
                    "max_label_available_at": train["label_available_at"].max(),
                    "max_label_end": train["label_end"].max(),
                    "max_training_date": train["date"].max(),
                    "scored_rows": len(eligible) if status == "fitted" else 0,
                    "seconds": perf_counter() - started,
                    "status": status,
                }
            )
    return result, pd.DataFrame(audit)
