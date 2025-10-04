import yaml
from typing import Dict
import numpy as np
from src.models.adaboost_model import ExoplanetAdaBoostModel
from src.models.ensemble import ExoplanetEnsemble
from src.data.loader import DataLoader

class ExoplanetDetectionPipeline:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path) if config_path else {}
        self.data_loader = DataLoader()
        self.ensemble = self.build_ensemble()

    def load_config(self, path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def build_ensemble(self):
        """Initialize ensemble with configured models"""
        models = []

        # AdaBoost/CatBoost base model
        adaboost_config = self.config.get('adaboost', {})
        adaboost_model = ExoplanetAdaBoostModel(adaboost_config)
        models.append(adaboost_model)

        # Add pretrained models if available
        if self.config.get('pretrained_model_path'):
            pretrained = ExoplanetAdaBoostModel()
            pretrained.load_model(self.config['pretrained_model_path'])
            models.append(pretrained)

        # Future: Add Random Forest or other models here

        strategy = self.config.get('ensemble_strategy', 'voting')
        return ExoplanetEnsemble(models, strategy=strategy)

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the ensemble"""
        self.ensemble.fit(x_train, y_train, x_val, y_val)

    def predict(self, X):
        """Get predictions for input features"""
        probabilities = self.ensemble.predict_proba(X)

        classes = ['false_positive', 'candidate', 'confirmed']
        results = []
        for i, probs in enumerate(probabilities):
            results.append({
                'predictions': dict(zip(classes, probs)),
                'classification': classes[np.argmax(probs)]
            })
        return results

    def process_batch(self, feature_data):
        """Process batch of feature data"""
        # Assumes features are already extracted
        return self.predict(feature_data)

    def save_model(self, path: str):
        """Save the primary model"""
        self.ensemble.models[0].save_model(path)

    def load_model(self, path: str):
        """Load a saved model"""
        self.ensemble.models[0].load_model(path)