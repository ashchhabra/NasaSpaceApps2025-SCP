from typing import List
import numpy as np

class ExoplanetEnsemble:
    def __init__(self, models: List, strategy='voting'):
        """
        Initialize ensemble with multiple models
        strategy: 'voting', 'stacking', or 'weighted'
        """
        self.models = models
        self.strategy = strategy
        self.weights = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Train all models in the ensemble"""
        print(f"Training ensemble with {len(self.models)} models ({self.strategy} strategy)") # test
        for model in self.models:
            if hasattr(model, 'train'):
                model.train(X, y, X_val, y_val)
            else:
                model.fit(X, y)

        if self.strategy == 'weighted':
            # Compute optimal weights based on validation performance
            self.weights = self.compute_optimal_weights(X, y)

    def predict_proba(self, X):
        """Combine predictions from all models"""
        predictions = np.array([model.predict_proba(X) for model in self.models])

        if self.strategy == 'voting':
            # Simple average
            return np.mean(predictions, axis=0)
        elif self.strategy == 'weighted':
            # Weighted average
            weighted_preds = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_preds += self.weights[i] * pred
            return weighted_preds
        elif self.strategy == 'stacking':
            # Use meta-learner (to be implemented)
            return self.meta_learner.predict_proba(predictions.reshape(len(X), -1))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict(self, X):
        """Get class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def compute_optimal_weights(self, X, y):
        """Simple weight optimization based on individual model performance"""
        # Placeholder - implement proper weight optimization
        # For now, return equal weights
        return np.ones(len(self.models)) / len(self.models)

    def add_model(self, model):
        """Add a new model to the ensemble"""
        self.models.append(model)