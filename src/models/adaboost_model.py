from catboost import CatBoostClassifier
from typing import Optional, Dict, Any
import numpy as np

class ExoplanetAdaBoostModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CatBoost with AdaBoost-like behavior optimized for 6 features"""
        default_config = {
            'iterations': 800,  # Reduced for 6 features to prevent overfitting
            'learning_rate': 0.05,  # Slower learning for better generalization
            'depth': 5,  # Optimal depth for 6 features
            'loss_function': 'MultiClass',
            'boosting_type': 'Plain',  # AdaBoost-like
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,  # Higher subsample for smaller feature space
            'random_seed': 42,
            'verbose': 100,  # Show progress every 100 iterations
            'early_stopping_rounds': 50,
            'task_type': 'CPU',  # or 'GPU' if available
            'auto_class_weights': 'Balanced',  # Handle class imbalance
            'l2_leaf_reg': 3.0,  # L2 regularization
            'min_data_in_leaf': 20,  # Prevent overfitting on small datasets
            'random_strength': 0.5,  # Add randomness for better generalization
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

    def get_feature_importance(self):
        """Get feature importances from the trained model"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def save_model(self, path: str):
        """Save model to file"""
        self.model.save_model(path)

    def load_model(self, path: str):
        """Load model from file"""
        self.model.load_model(path)