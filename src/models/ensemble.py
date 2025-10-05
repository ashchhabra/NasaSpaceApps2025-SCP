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

        print(f"\n{'='*60}")
        print(f"ENSEMBLE CONFIGURATION")
        print(f"{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"Number of models: {len(models)}")
        for i, model in enumerate(models):
            model_name = type(model).__name__.replace('Exoplanet', '').replace('Model', '')
            print(f"  Model {i+1}: {model_name}")
        print(f"{'='*60}\n")

    def fit(self, X, y, X_val=None, y_val=None):
        """Train all models in the ensemble"""
        for model in self.models:
            if hasattr(model, 'train'):
                model.train(X, y, X_val, y_val)
            else:
                model.fit(X, y)

        if self.strategy == 'weighted':
            # Compute optimal weights based on validation performance
            self.weights = self.compute_optimal_weights(X_val, y_val)

    def predict_proba(self, X):
        """Combine predictions from all models"""
        predictions = np.array([model.predict_proba(X) for model in self.models])

        if self.strategy == 'voting':
            # Simple average - all models have equal weight
            return np.mean(predictions, axis=0)
        elif self.strategy == 'weighted':
            # Weighted average - models weighted by validation performance
            if self.weights is None:
                raise ValueError("Weights not computed. Did you call fit()?")
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

    def compute_optimal_weights(self, X_val, y_val):
        """Compute weights based on validation accuracy"""
        if X_val is None or y_val is None:
            print("Warning: No validation data provided, using equal weights")
            return np.ones(len(self.models)) / len(self.models)

        weights = []
        print("\nComputing optimal weights based on validation performance:")

        for i, model in enumerate(self.models):
            model_name = type(model).__name__.replace('Exoplanet', '').replace('Model', '')
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            weights.append(accuracy)
            print(f"  {model_name}: accuracy = {accuracy:.4f}")

        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()

        print(f"\nNormalized weights: {weights}")
        for i, model in enumerate(self.models):
            model_name = type(model).__name__.replace('Exoplanet', '').replace('Model', '')
            print(f"  {model_name}: {weights[i]:.4f}")

        return weights

    def add_model(self, model):
        """Add a new model to the ensemble"""
        self.models.append(model)