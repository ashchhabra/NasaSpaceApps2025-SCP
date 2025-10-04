import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class DataLoader:
    # Expected feature columns in order
    EXPECTED_FEATURES = [
        'planet_radii',
        'transit_depth',
        'days',
        'stars_radii',
        'earth_flux',
        'star_temp'
    ]

    # Class labels mapping
    CLASS_LABELS = {
        'confirmed': 0,
        'candidate': 1,
        'false_positive': 2
    }

    def __init__(self):
        """Enhanced loader with feature validation and normalization"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['confirmed', 'candidate', 'false_positive'])

    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and reorder features to match expected schema"""
        # Check if all expected features are present
        missing_features = set(self.EXPECTED_FEATURES) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Reorder columns to match expected order
        feature_cols = [col for col in self.EXPECTED_FEATURES if col in df.columns]
        other_cols = [col for col in df.columns if col not in self.EXPECTED_FEATURES]

        return df[feature_cols + other_cols]

    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)

    def encode_labels(self, labels: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Convert string labels to numeric"""
        if isinstance(labels, (pd.Series, pd.DataFrame)):
            labels = labels.values

        # Handle string labels
        if isinstance(labels[0], str):
            # Map using our predefined mapping
            encoded = np.array([self.CLASS_LABELS.get(label, -1) for label in labels])
            if -1 in encoded:
                unknown_labels = set(labels) - set(self.CLASS_LABELS.keys())
                raise ValueError(f"Unknown labels found: {unknown_labels}")
            return encoded
        else:
            # Already numeric
            return np.array(labels, dtype=int)

    def load_features(self, data: Union[str, np.ndarray, pd.DataFrame],
                     normalize: bool = True, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load feature data with validation and normalization
        Returns: (features, labels) tuple
        """
        if isinstance(data, str):
            # Load from CSV file
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            # Convert to DataFrame for easier handling
            if data.shape[1] == len(self.EXPECTED_FEATURES) + 1:
                columns = self.EXPECTED_FEATURES + ['label']
            else:
                columns = [f'feature_{i}' for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=columns)
        else:
            raise ValueError("Unsupported data format")

        # Validate features if columns match expected names
        if all(feat in df.columns for feat in self.EXPECTED_FEATURES):
            df = self.validate_features(df)
            features = df[self.EXPECTED_FEATURES].values
        else:
            # Assume first n columns are features, last is label
            features = df.iloc[:, :-1].values

        # Get labels (last column)
        labels = df.iloc[:, -1]
        encoded_labels = self.encode_labels(labels)

        # Check for NaN values
        if np.any(np.isnan(features)):
            print("Warning: NaN values found in features. Filling with mean.")
            features = np.nan_to_num(features, nan=np.nanmean(features[~np.isnan(features)]))

        # Normalize features if requested
        if normalize:
            features = self.normalize_features(features, fit=fit_scaler)

        return features, encoded_labels

    def save_scaler(self, path: str):
        """Save the fitted scaler for later use"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, path: str):
        """Load a previously fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)

    def load_batch(self, file_paths: List[str]) -> np.ndarray:
        """Load batch of pre-processed files"""
        all_features = []
        for path in file_paths:
            # Assume each file contains extracted features
            features = np.load(path) if path.endswith('.npy') else pd.read_csv(path).values
            all_features.append(features)
        return np.vstack(all_features)