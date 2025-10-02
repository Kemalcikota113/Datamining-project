"""
Quality Measures Implementation
Simplified implementations using sklearn with proper error handling.
"""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .core import QualityMeasure


def _validate_clustering_data(clustering_result, dataset):
    """
    Helper function to validate and prepare clustering data.
    
    Returns:
        tuple: (valid_data, valid_labels) or (None, None) if invalid
    """
    data_points = dataset.get_data_points()
    labels = clustering_result.get_labels()
    
    # Filter out noise points (label -1) for DBSCAN
    valid_mask = labels >= 0
    if np.sum(valid_mask) < 2:
        return None, None
    
    valid_data = data_points[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Need at least 2 clusters for meaningful quality scores
    if len(np.unique(valid_labels)) < 2:
        return None, None
    
    return valid_data, valid_labels


class SilhouetteScore(QualityMeasure):
    """
    Silhouette score using sklearn implementation.
    Range: [-1, 1], higher is better.
    """
    
    def evaluate(self, clustering_result, dataset):
        valid_data, valid_labels = _validate_clustering_data(clustering_result, dataset)
        if valid_data is None:
            return -1.0  # Worst possible score
        
        return silhouette_score(valid_data, valid_labels)


class CalinskiHarabaszScore(QualityMeasure):
    """
    Calinski-Harabasz score using sklearn implementation.
    Range: [0, +inf), higher is better.
    """
    
    def evaluate(self, clustering_result, dataset):
        valid_data, valid_labels = _validate_clustering_data(clustering_result, dataset)
        if valid_data is None:
            return 0.0  # Worst possible score
        
        return calinski_harabasz_score(valid_data, valid_labels)


class DaviesBouldinScore(QualityMeasure):
    """
    Davies-Bouldin score using sklearn implementation.
    Range: [0, +inf), lower is better.
    """
    
    def evaluate(self, clustering_result, dataset):
        valid_data, valid_labels = _validate_clustering_data(clustering_result, dataset)
        if valid_data is None:
            return float('inf')  # Worst possible score
        
        return davies_bouldin_score(valid_data, valid_labels)
