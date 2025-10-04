from catboost import CatBoostClassifier
from typing import Optional, Dict, Any
import numpy as np

class ExoplanetAdaBoostModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CatBoost with AdaBoost-like behavior"""
        default_config = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'MultiClass',
            'boosting_type': 'Plain',  # AdaBoost-like
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.66,
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50,
            'task_type': 'CPU'  # or 'GPU' if available
        }
        self.config = {**default_config, **(config or {})}
        self.model = CatBoostClassifier(**self.config)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional validation set"""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            use_best_model=True if eval_set else False
        )
        return self

    def predict_proba(self, X):
        """Return probability distribution over classes"""
        return self.model.predict_proba(X)

    def predict(self, X):
        """Return class predictions"""
        return self.model.predict(X)

    def save_model(self, path: str):
        """Save model to file"""
        self.model.save_model(path)

    def load_model(self, path: str):
        """Load model from file"""
        self.model.load_model(path)