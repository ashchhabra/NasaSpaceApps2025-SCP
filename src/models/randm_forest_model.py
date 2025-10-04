from typing import Optional, Dict, Any, Union, TYPE_CHECKING, TypeAlias
import numpy as np
from numpy.typing import NDArray
import pandas as pd

ArrayLike: TypeAlias = Union[NDArray[Any], pd.DataFrame]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib # saving/loading trained models


class ExoplanetRandomForestModel:
    """
    Random Forest model wrapper with the public API 
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RandomForest with sensible defaults for tabular KOI data."""
        default_config: Dict[str, Any] = {
            "n_estimators": 800,        # more trees => more stable
            "max_depth": None,          # let trees grow; control with min_samples_leaf
            "min_samples_split": 2,
            "min_samples_leaf": 1,      # try 2–4 to reduce variance
            "max_features": "sqrt",     # classic setting for classification
            "bootstrap": True,
            "class_weight": None,       # or "balanced" if classes are imbalanced
            "random_state": 42,
            "n_jobs": -1,               # use all cores
            "verbose": 0,
        }
        self.config = {**default_config, **(config or {})}
        self.model = RandomForestClassifier(**self.config)

    def train(
        self,
        X_train: ArrayLike,
        y_train: np.ndarray,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ExoplanetRandomForestModel":
        """
        Fit the Random Forest.
        NOTE: scikit-learn RF has no early stopping; if X_val/y_val are provided,
              we just evaluate after fit and return metrics (printed).
        """
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            y_prob = self.model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average="binary", zero_division=0
            )
            print(
                f"[RF] Validation — Acc: {acc:.3f} | F1: {f1:.3f} | "
                f"Precision: {p:.3f} | Recall: {r:.3f}"
            )
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        joblib.dump({"model": self.model, "config": self.config}, path) # save

    def load_model(self, path: str) -> None:
        payload = joblib.load(path) # load
        self.model = payload["model"]
        self.config = payload.get("config", self.config)