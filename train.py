from src.pipeline import ExoplanetDetectionPipeline
from src.data.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
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

    # Load training data (assumed to be pre-processed features)
    data_path = 'data/training_features.csv' if len(sys.argv) < 3 else sys.argv[2]

    try:
        X, y = loader.load_features(data_path)
        print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        print(f"Data file not found at {data_path}")
        print("Creating dummy data for testing...")
        # Create dummy data for testing
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        # 3 classes: 0=confirmed, 1=candidate, 2=false_positive
        y = np.random.randint(0, 3, n_samples)

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
    print("\nEvaluating...")
    predictions = pipeline.predict(X_test)
    y_pred = [pred['predictions'].get('confirmed', 0) * 0 +
              pred['predictions'].get('candidate', 0) * 1 +
              pred['predictions'].get('false_positive', 0) * 2
              for pred in predictions]

    # Convert probabilities to class indices
    y_pred = []
    class_map = {'confirmed': 0, 'candidate': 1, 'false_positive': 2}
    for pred in predictions:
        y_pred.append(class_map[pred['classification']])

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['confirmed', 'candidate', 'false_positive']))

    # Save model
    model_path = 'models/adaboost_exoplanet.cbm'
    print(f"\nSaving model to {model_path}")
    import os
    os.makedirs('models', exist_ok=True)
    pipeline.save_model(model_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()