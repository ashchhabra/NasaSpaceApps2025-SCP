from src.pipeline import ExoplanetDetectionPipeline
from src.data.loader import DataLoader
import numpy as np
import pandas as pd
import sys
import json

def main():
    # Initialize pipeline
    pipeline = ExoplanetDetectionPipeline()

    # Load model
    model_path = 'models/adaboost_exoplanet.cbm' if len(sys.argv) < 2 else sys.argv[1]
    print(f"Loading model from {model_path}")

    try:
        pipeline.load_model(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please train the model first using: python train.py")
        return

    # Initialize data loader and load scaler
    loader = DataLoader()
    try:
        loader.load_scaler('models/feature_scaler.pkl')
        print("Loaded feature scaler")
    except FileNotFoundError:
        print("Warning: No scaler found. Features will not be normalized.")

    # Load data for prediction
    data_path = 'data/test_features.csv' if len(sys.argv) < 3 else sys.argv[2]

    try:
        # Check if input is JSON (single sample) or CSV (batch)
        if data_path.endswith('.json'):
            # Load single sample from JSON
            with open(data_path, 'r') as f:
                sample = json.load(f)

            # Create DataFrame with expected columns
            df = pd.DataFrame([sample])
            X, _ = loader.load_features(df, normalize=True, fit_scaler=False)
            print(f"Loaded 1 sample for prediction")
        else:
            # Load batch from CSV
            df = pd.read_csv(data_path)

            # Add dummy label column if not present (for compatibility)
            if 'label' not in df.columns and len(df.columns) == 6:
                df['label'] = 'unknown'

            X, _ = loader.load_features(df, normalize=True, fit_scaler=False)
            print(f"Loaded {X.shape[0]} samples for prediction")

        # Validate feature count
        if X.shape[1] != 6:
            print(f"Warning: Expected 6 features but got {X.shape[1]}")
            print("Expected: planet_radii, transit_depth, days, stars_radii, earth_flux, star_temp")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nGenerating sample data for demo...")
        # Generate realistic sample data matching our schema
        X = np.column_stack([
            [1.2, 0.8, 1.5, 2.0, 0.5],  # planet_radii
            [0.003, 0.001, 0.005, 0.008, 0.0002],  # transit_depth
            [365.25, 10.5, 50.0, 200.0, 5.0],  # days
            [1.0, 0.9, 1.1, 1.5, 0.7],  # stars_radii
            [1.0, 0.5, 2.0, 0.3, 4.0],  # earth_flux
            [5778, 4000, 6500, 3500, 7000]  # star_temp
        ]).T

        # Normalize if scaler is available
        try:
            X = loader.normalize_features(X, fit=False)
        except:
            print("Using raw features (no normalization)")

    # Predict
    predictions = pipeline.predict(X)

    # Display results
    print("\n" + "="*60)
    print("EXOPLANET DETECTION RESULTS")
    print("="*60)

    for i, pred in enumerate(predictions):
        print(f"\nSample {i+1}:")
        print(f"  Classification: {pred['classification'].upper()}")

        # Show confidence level
        max_prob = max(pred['predictions'].values())
        if max_prob > 0.8:
            confidence_level = "VERY HIGH"
        elif max_prob > 0.6:
            confidence_level = "HIGH"
        elif max_prob > 0.4:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"
        print(f"  Confidence Level: {confidence_level}")

        print(f"  Probability Scores:")
        for class_name, prob in pred['predictions'].items():
            bar = '█' * int(prob * 20)
            print(f"    {class_name:15s}: {prob:.3f} {bar}")

    # Save results
    results_df = pd.DataFrame([{
        'sample_id': i + 1,
        'classification': pred['classification'],
        'confidence': max(pred['predictions'].values()),
        'confirmed_prob': pred['predictions'].get('confirmed', 0),
        'candidate_prob': pred['predictions'].get('candidate', 0),
        'false_positive_prob': pred['predictions'].get('false_positive', 0)
    } for i, pred in enumerate(predictions)])

    output_path = 'predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")

    # Also save as JSON for easier processing
    json_output = 'predictions.json'
    with open(json_output, 'w') as f:
        json.dump([{
            'sample_id': i + 1,
            'classification': pred['classification'],
            'probabilities': pred['predictions']
        } for i, pred in enumerate(predictions)], f, indent=2)
    print(f"Results also saved to {json_output}")

if __name__ == "__main__":
    main()