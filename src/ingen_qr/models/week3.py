"""Week 3 regressors with train-only preprocessing."""

from importlib import import_module
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> None: ...

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]: ...

    def get_params(self) -> dict[str, Any]: ...


class Regressor:
    def __init__(self, name: str, params: dict[str, Any], seed: int) -> None:
        self.params = {**params, "random_state": seed}
        model_options = dict(self.params)
        if name == "mlp":
            model_options["hidden_layer_sizes"] = (model_options.pop("hidden_width"),)
        module, class_name = {
            "ridge": ("sklearn.linear_model", "Ridge"),
            "elastic_net": ("sklearn.linear_model", "ElasticNet"),
            "random_forest": ("sklearn.ensemble", "RandomForestRegressor"),
            "lightgbm": ("lightgbm", "LGBMRegressor"),
            "adaboost": ("sklearn.ensemble", "AdaBoostRegressor"),
            "mlp": ("sklearn.neural_network", "MLPRegressor"),
        }[name]
        estimator = getattr(import_module(module), class_name)(**model_options)
        steps = [
            (
                "impute",
                import_module("sklearn.impute")
                .SimpleImputer(strategy="median", keep_empty_features=True)
                .set_output(transform="pandas"),
            )
        ]
        if name in {"ridge", "elastic_net", "mlp"}:
            steps.append(
                ("scale", import_module("sklearn.preprocessing").StandardScaler())
            )
        steps.append(("model", estimator))
        self.pipeline = import_module("sklearn.pipeline").Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> None:
        self.pipeline.fit(X.replace([np.inf, -np.inf], np.nan).to_numpy(), y.to_numpy())

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        values = X.replace([np.inf, -np.inf], np.nan).to_numpy()
        return np.asarray(self.pipeline.predict(values), dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)
