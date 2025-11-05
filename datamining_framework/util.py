import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .core import Dataset


def load_csv(filepath, **kwargs):

    df = pd.read_csv(filepath, **kwargs)
    return Dataset(df)


def normalize_dataset(dataset, method='standard', **kwargs):

    if method == 'standard':
        scaler = StandardScaler(**kwargs)
    elif method == 'minmax':
        scaler = MinMaxScaler(**kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'standard' or 'minmax'.")
    
    # Get original data
    data_points = dataset.get_data_points()
    features = dataset.get_features()
    
    # Fit and transform
    normalized_data = scaler.fit_transform(data_points)
    
    # Create new DataFrame with normalized data
    normalized_df = pd.DataFrame(normalized_data, columns=features)
    
    return Dataset(normalized_df)


def save_csv(dataset, filepath, **kwargs):

    data_points = dataset.get_data_points()
    features = dataset.get_features()
    
    df = pd.DataFrame(data_points, columns=features)
    df.to_csv(filepath, **kwargs)


def get_dataset_info(dataset):

    data_points = dataset.get_data_points()
    
    info = {
        'shape': dataset.get_shape(),
        'n_samples': dataset.get_shape()[0],
        'n_features': dataset.get_shape()[1],
        'features': dataset.get_features(),
        'mean': np.mean(data_points, axis=0).tolist(),
        'std': np.std(data_points, axis=0).tolist(),
        'min': np.min(data_points, axis=0).tolist(),
        'max': np.max(data_points, axis=0).tolist()
    }
    
    return info


def split_features_labels(dataset, label_column):

    data_points = dataset.get_data_points()
    features = dataset.get_features()
    
    if isinstance(label_column, str):
        label_idx = features.index(label_column)
    else:
        label_idx = label_column
    
    # Extract labels
    labels = data_points[:, label_idx]
    
    # Extract features (all columns except label)
    feature_data = np.delete(data_points, label_idx, axis=1)
    feature_names = [f for i, f in enumerate(features) if i != label_idx]
    
    # Create new dataset with only features
    feature_df = pd.DataFrame(feature_data, columns=feature_names)
    features_dataset = Dataset(feature_df)
    
    return features_dataset, labels
