from .core import Dataset, ClusteringResult
from .distance_measures import EuclideanDistance, ManhattanDistance, CosineDistance
from .clustering_techniques import KMeansClustering, DBSCANClustering, HierarchicalClustering
from .quality_measures import SilhouetteScore, CalinskiHarabaszScore, DaviesBouldinScore

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
    
    # Convenience functions
    'load_dataset',
    'get_all_distance_measures',
    'get_all_clustering_techniques', 
    'get_all_quality_measures'
]

# Convenience functions for easy access
def load_dataset(data):
    return Dataset(data)

def get_all_distance_measures():
    return {
        'euclidean': EuclideanDistance,
        'manhattan': ManhattanDistance,
        'cosine': CosineDistance
    }

def get_all_clustering_techniques():
    return {
        'kmeans': KMeansClustering,
        'dbscan': DBSCANClustering,
        'hierarchical': HierarchicalClustering
    }

def get_all_quality_measures():
    return {
        'silhouette': SilhouetteScore,
        'calinski_harabasz': CalinskiHarabaszScore,
        'davies_bouldin': DaviesBouldinScore
    }