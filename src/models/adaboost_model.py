from catboost import CatBoostClassifier
from typing import Optional, Dict, Any
import numpy as np

class ExoplanetAdaBoostModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CatBoost with AdaBoost-like behavior optimized for 6 features"""
        default_config = {
            'iterations': 1500,  # Increased for better learning with 16k samples
            'learning_rate': 0.035,  # Slightly slower for more iterations
            'depth': 6,  # Good depth for 6 features
            'loss_function': 'MultiClass',
            'boosting_type': 'Plain',  # AdaBoost-like
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.85,  # Subsample ratio for Bernoulli
            'random_seed': 42,
            'verbose': 50,  # Show progress every 50 iterations
            'metric_period': 25,  # Calculate metrics every 25 iterations
            'early_stopping_rounds': None,  # Disable early stopping temporarily
            'task_type': 'CPU',  # or 'GPU' if available
            'auto_class_weights': 'Balanced',  # Handle class imbalance
            'od_type': 'None',  # Disable overfitting detector
            'l2_leaf_reg': 2.0,  # Reduced L2 regularization
            'min_data_in_leaf': 10,  # Allow smaller leaves with more data
            'random_strength': 1.0,  # Randomness for scoring
            'border_count': 128,  # Number of splits for numerical features
            'grow_policy': 'SymmetricTree',  # Tree growing policy
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
            use_best_model=True if eval_set else False,
            plot=True  # Show convergence plot
        )

        # Print convergence info
        if hasattr(self.model, 'best_iteration_'):
            print(f"\n✓ Model converged at iteration {self.model.best_iteration_}")
            print(f"  (Early stopped after {self.model.best_iteration_ + self.config['early_stopping_rounds']} iterations)")
        else:
            print(f"\n✓ Model completed all {self.config['iterations']} iterations without early stopping")

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

    def get_training_history(self):
        """Get training history including loss values"""
        if hasattr(self.model, 'evals_result_'):
            return self.model.evals_result_
        return None

    def save_model(self, path: str):
        """Save model to file"""
        self.model.save_model(path)

    def load_model(self, path: str):
        """Load model from file"""
        self.model.load_model(path)