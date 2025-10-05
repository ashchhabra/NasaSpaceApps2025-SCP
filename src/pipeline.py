import yaml
from typing import Dict
import numpy as np
from src.models.adaboost_model import ExoplanetAdaBoostModel
from src.models.randm_forest_model import ExoplanetRandomForestModel
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

        # random forest base model
        randm_config = self.config.get('randm_forest', {})
        randm_forest_model = ExoplanetRandomForestModel(randm_config)
        models.append(randm_forest_model)

        # Add pretrained models if available
        '''
        if self.config.get('pretrained_model_path'):
            pretrained = ExoplanetRandomForestModel() 
            pretrained.load_model(self.config['pretrained_model_path'])
            models.append(pretrained)
        '''
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
        #self.ensemble.models[0].save_model(path)
        """
        Save all trained models in the ensemble.
        Example:
            base_path='models/ensemble_exoplanet'
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = []

        for model in self.ensemble.models:
            model_name = model.__class__.__name__.replace("Exoplanet", "").replace("Model", "").lower()
            model_path = f"{path}_{model_name}.pkl"
            model.save_model(model_path)
            print(f"Saved {model_name} model to {model_path}")
            meta.append({"name": model_name, "path": model_path})

        # Save ensemble metadata
        meta_path = f"{path}_meta.pkl"
        joblib.dump({
            "ensemble_strategy": self.ensemble.strategy,
            "models": meta
        }, meta_path)
        print(f"Saved ensemble metadata to {meta_path}")

    def load_model(self, path: str):
        """Load a saved model"""
        self.ensemble.models[0].load_model(path)