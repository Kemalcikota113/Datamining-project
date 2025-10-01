"""
Data Mining Framework - Main API
Simplified framework focused on Project 1 requirements with extensible design.
"""

from .core import Dataset, ClusteringResult
from .distance_measures import EuclideanDistance, ManhattanDistance, CosineDistance
from .clustering_techniques import KMeansClustering, DBSCANClustering, HierarchicalClustering
from .quality_measures import SilhouetteScore, CalinskiHarabaszScore, DaviesBouldinScore, InertiaScore

# Version info
__version__ = "2.0.0"
__author__ = "Kemal"

# Export main components for easy access
__all__ = [
    # Core components
    'Dataset',
    'ClusteringResult',
    
    # Distance measures
    'EuclideanDistance',
    'ManhattanDistance', 
    'CosineDistance',
    
    # Clustering techniques
    'KMeansClustering',
    'DBSCANClustering',
    'HierarchicalClustering',
    
    # Quality measures
    'SilhouetteScore',
    'CalinskiHarabaszScore',
    'DaviesBouldinScore',
    'InertiaScore',
    
    # Convenience functions
    'load_dataset',
    'get_all_distance_measures',
    'get_all_clustering_techniques', 
    'get_all_quality_measures'
]

# Convenience functions for easy access
def load_dataset(data):
    """
    Load dataset from various sources.
    
    Args:
        data: pandas DataFrame, numpy array, or file path
        
    Returns:
        Dataset object
    """
    return Dataset(data)

def get_all_distance_measures():
    """
    Get all available distance measures.
    
    Returns:
        dict: Dictionary of distance measure name -> class
    """
    return {
        'euclidean': EuclideanDistance,
        'manhattan': ManhattanDistance,
        'cosine': CosineDistance
    }

def get_all_clustering_techniques():
    """
    Get all available clustering techniques.
    
    Returns:
        dict: Dictionary of clustering technique name -> class
    """
    return {
        'kmeans': KMeansClustering,
        'dbscan': DBSCANClustering,
        'hierarchical': HierarchicalClustering
    }

def get_all_quality_measures():
    """
    Get all available quality measures.
    
    Returns:
        dict: Dictionary of quality measure name -> class
    """
    return {
        'silhouette': SilhouetteScore,
        'calinski_harabasz': CalinskiHarabaszScore,
        'davies_bouldin': DaviesBouldinScore,
        'inertia': InertiaScore
    }
