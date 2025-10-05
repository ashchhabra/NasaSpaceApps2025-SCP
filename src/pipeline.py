import yaml
from typing import Dict
import numpy as np
from src.models.adaboost_model import ExoplanetAdaBoostModel
from src.models.random_forest_model import ExoplanetRandomForestModel
from src.models.ensemble import ExoplanetEnsemble
from src.data.loader import DataLoader
import os
import joblib

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

        # Random Forest model
        random_forest_config = self.config.get('random_forest', {})
        rf_model = ExoplanetRandomForestModel(random_forest_config)
        models.append(rf_model)

        # Add pretrained models if available
    
        if self.config.get('pretrained_model_path'):
            pretrained = ExoplanetRandomForestModel() 
            pretrained.load_model(self.config['pretrained_model_path'])
            models.append(pretrained)
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
        """Save all models in the ensemble"""
        import os
        base_dir = os.path.dirname(path)

        # Save CatBoost model (model 0)
        catboost_path = os.path.join(base_dir, 'adaboost_exoplanet.cbm')
        self.ensemble.models[0].save_model(catboost_path)

        # Save Random Forest model (model 1)
        rf_path = os.path.join(base_dir, 'random_forest.pkl')
        self.ensemble.models[1].save_model(rf_path)

    def load_model(self, path: str):
        """Load all models in the ensemble"""
        import os
        base_dir = os.path.dirname(path)

        # Load CatBoost model (model 0)
        catboost_path = os.path.join(base_dir, 'adaboost_exoplanet.cbm')
        self.ensemble.models[0].load_model(catboost_path)

        # Load Random Forest model (model 1)
        rf_path = os.path.join(base_dir, 'random_forest.pkl')
        self.ensemble.models[1].load_model(rf_path)