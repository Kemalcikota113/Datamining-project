"""
Utility Functions
Helper functions for data loading and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .core import Dataset


def load_csv(filepath, **kwargs):
    """
    Load a dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pandas.read_csv()
        
    Returns:
        Dataset: Dataset object containing the loaded data
        
    Example:
        >>> dataset = load_csv('data.csv')
        >>> dataset = load_csv('data.csv', sep=';', header=0)
    """
    df = pd.read_csv(filepath, **kwargs)
    return Dataset(df)


def normalize_dataset(dataset, method='standard', **kwargs):
    """
    Normalize/standardize a dataset using sklearn scalers.
    
    Args:
        dataset (Dataset): Dataset to normalize
        method (str): Normalization method - 'standard' (z-score) or 'minmax' (0-1 scaling)
        **kwargs: Additional arguments to pass to the scaler
        
    Returns:
        Dataset: New normalized dataset
        
    Example:
        >>> normalized = normalize_dataset(dataset, method='standard')
        >>> normalized = normalize_dataset(dataset, method='minmax')
    """
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
    """
    Save a dataset to a CSV file.
    
    Args:
        dataset (Dataset): Dataset to save
        filepath (str): Path where to save the CSV file
        **kwargs: Additional arguments to pass to pandas.to_csv()
        
    Example:
        >>> save_csv(dataset, 'output.csv')
        >>> save_csv(dataset, 'output.csv', index=False)
    """
    data_points = dataset.get_data_points()
    features = dataset.get_features()
    
    df = pd.DataFrame(data_points, columns=features)
    df.to_csv(filepath, **kwargs)


def get_dataset_info(dataset):
    """
    Get summary information about a dataset.
    
    Args:
        dataset (Dataset): Dataset to analyze
        
    Returns:
        dict: Dictionary containing dataset statistics
        
    Example:
        >>> info = get_dataset_info(dataset)
        >>> print(info['shape'])
    """
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
    """
    Split dataset into features and labels.
    
    Args:
        dataset (Dataset): Dataset containing features and labels
        label_column (str or int): Name or index of the label column
        
    Returns:
        tuple: (features_dataset, labels_array)
        
    Example:
        >>> features, labels = split_features_labels(dataset, 'target')
        >>> features, labels = split_features_labels(dataset, -1)
    """
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
