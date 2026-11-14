import numpy as np
import pandas as pd
import pytest

from ingen_qr.evaluation.signal_metrics import monthly_metrics, newey_west
from ingen_qr.models.linear import OLS_FEATURES, ols_predict


def test_newey_west_matches_independent_bartlett_covariance() -> None:
    values = np.array([0.2, -0.1, 0.15, 0.05, -0.03, 0.11, 0.04, 0.09] * 5)
    n, lag = len(values), 6
    errors = values - values.mean()
    long_run = errors @ errors / n
    long_run += 2 * sum(
        (1 - k / (lag + 1)) * (errors[k:] @ errors[:-k]) / n for k in range(1, lag + 1)
    )
    standard_error = np.sqrt(long_run / n * n / (n - 1))
    assert newey_west(pd.Series(values)) == pytest.approx(
        values.mean() / standard_error
    )


def test_ic_rank_ic_decile_spread_and_rank_hit_rate_by_hand() -> None:
    rows = pd.DataFrame(
        {
            "model": "momentum",
            "date": pd.Timestamp("2020-01-02"),
            "prediction": np.arange(10.0),
            "label_excess_return": np.arange(10.0, 0.0, -1),
        }
    )
    result = monthly_metrics(rows).iloc[0]
    assert result["ic"] == pytest.approx(-1)
    assert result["rank_ic"] == pytest.approx(-1)
    assert result["decile_spread"] == -9
    assert result["rank_hit_rate"] == 0
    assert pd.isna(result["hit_rate"])


def test_ols_three_characteristics_recover_known_coefficients() -> None:
    random = np.random.default_rng(4)
    train = pd.DataFrame(random.normal(size=(30, 3)), columns=OLS_FEATURES)
    target = (
        1
        + 2 * train[OLS_FEATURES[0]]
        + 3 * train[OLS_FEATURES[1]]
        - 4 * train[OLS_FEATURES[2]]
    )
    test = pd.DataFrame([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]], columns=OLS_FEATURES)
    assert ols_predict(train, target, test) == pytest.approx([-3.0, 8.0])
