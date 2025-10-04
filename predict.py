
from src.pipeline import ExoplanetDetectionPipeline
import numpy as np
import pandas as pd
import sys

def main():
    # Load trained pipeline
    config_path = 'configs/model_config.yaml' if len(sys.argv) < 2 else sys.argv[1]
    model_path = 'models/adaboost_exoplanet.cbm' if len(sys.argv) < 3 else sys.argv[2]
    data_path = None if len(sys.argv) < 4 else sys.argv[3]

    print("Loading model...")
    pipeline = ExoplanetDetectionPipeline(config_path)

    try:
        pipeline.load_model(model_path)
        print(f"Model loaded from {model_path}")
    except:
        print(f"Could not load model from {model_path}")
        print("Using untrained model for demonstration")

    # Process new data
    if data_path:
        # Load from file
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
            features = data.values if data.shape[1] > 1 else data.values.reshape(-1, 1)
        elif data_path.endswith('.npy'):
            features = np.load(data_path)
        else:
            print(f"Unsupported file format: {data_path}")
            return
    else:
        # Generate dummy data for demonstration
        print("\nNo data provided. Using dummy data for demonstration...")
        np.random.seed(123)
        features = np.random.randn(5, 10)  # 5 samples, 10 features

    print(f"\nProcessing {features.shape[0]} samples...")
    results = pipeline.predict(features)

    print("\n" + "="*60)
    print("EXOPLANET DETECTION RESULTS")
    print("="*60)

    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Classification: {result['classification'].upper()}")
        print(f"  Confidence Scores:")
        for class_name, prob in result['predictions'].items():
            print(f"    {class_name:15s}: {prob:.2%}")
        print("-" * 40)

if __name__ == "__main__":
    main()