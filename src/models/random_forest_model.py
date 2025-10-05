from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Dict, Any
import numpy as np
import pickle

class ExoplanetRandomForestModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Random Forest classifier optimized for 6 features and 3 classes"""
        default_config = {
            'n_estimators': 400,  # Good balance of performance and speed
            'max_depth': 10,  # Moderate depth for 6 features
            'min_samples_split': 5,  # Prevent overfitting
            'min_samples_leaf': 3,  # Handle class imbalance
            'max_features': 'sqrt',  # Standard for classification
            'class_weight': 'balanced',  # Critical for imbalanced classes
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'bootstrap': True,  # Standard bagging
            'oob_score': True,  # Out-of-bag score for validation
            'verbose': 1
        }
        self.config = {**default_config, **(config or {})}
        self.model = RandomForestClassifier(**self.config)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Random Forest model"""
        print(f"\nTraining Random Forest with {self.config['n_estimators']} trees...")
        print(f"  Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Class distribution: {np.bincount(y_train)}")

        self.model.fit(X_train, y_train)

        # Print training info
        print(f"✓ Random Forest training complete")
        if hasattr(self.model, 'oob_score_'):
            print(f"  Out-of-bag score: {self.model.oob_score_:.4f}")

        # Validation accuracy if provided
        if X_val is not None and y_val is not None:
            val_accuracy = self.model.score(X_val, y_val)
            print(f"  Validation accuracy: {val_accuracy:.4f}")

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
        """Save model to file using pickle"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def fit(self, X, y):
        """Scikit-learn compatible fit method"""
        return self.train(X, y)
