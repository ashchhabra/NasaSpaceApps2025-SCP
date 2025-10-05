from src.pipeline import ExoplanetDetectionPipeline
from src.data.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import os
import sys

def main():
    # Initialize pipeline
    config_path = 'configs/model_config.yaml' if len(sys.argv) < 2 else sys.argv[1]

    try:
        pipeline = ExoplanetDetectionPipeline(config_path)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using default configuration")
        pipeline = ExoplanetDetectionPipeline()

    loader = DataLoader()

    # Load training data - expects 6 features + label
    data_path = 'src/data/consolidated.csv' if len(sys.argv) < 3 else sys.argv[2]

    try:
        # Load with normalization and fit the scaler
        X, y = loader.load_features(data_path, normalize=True, fit_scaler=True)
        print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")

        # Validate feature count
        if X.shape[1] != 6:
            raise ValueError(f"Expected 6 features but got {X.shape[1]}. "
                           "Expected: planet_radii, transit_depth, days, stars_radii, earth_flux, star_temp")

        # Save scaler for prediction use
        os.makedirs('models', exist_ok=True)
        loader.save_scaler('models/feature_scaler.pkl')
        print("Feature scaler saved to models/feature_scaler.pkl")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure consolidated.csv exists at src/data/consolidated.csv")
        sys.exit(1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Train
    print("Training ensemble...")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    pipeline.train(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\nEvaluating on test set...")
    predictions = pipeline.predict(X_test)

    # Convert probabilities to class indices
    y_pred = []
    class_map = {'false_positive': 0, 'candidate': 1, 'confirmed': 2}
    for pred in predictions:
        y_pred.append(class_map[pred['classification']])

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['false_positive', 'candidate', 'confirmed']))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("              Predicted")
    print("             FalsePos  Cand  Conf")
    labels = ['FalsePos', 'Candidate', 'Confirmed']
    for i, label in enumerate(labels):
        print(f"Actual {label:9s}: {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")

    # Feature importance (if available)
    try:
        if hasattr(pipeline.ensemble.models[0].model, 'feature_importances_'):
            importances = pipeline.ensemble.models[0].model.feature_importances_
            feature_names = loader.EXPECTED_FEATURES
            print("\nFeature Importances:")
            for name, importance in zip(feature_names, importances):
                print(f"  {name:15s}: {importance:.4f}")
    except:
        pass

    # Save model
    #model_path = 'models/adaboost_exoplanet.cbm'
    #model_path = 'models/randm_forest_exoplanet.joblib' # commented
    model_path = 'models/ensemble_exoplanet'
    print(f"\nSaving model to {model_path}")
    pipeline.save_model(model_path)
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    main()