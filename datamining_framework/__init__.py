from .core import Dataset, ClusteringResult, DRResult
from .distance_measures import EuclideanDistance, ManhattanDistance, CosineDistance
from .clustering_techniques import KMeansClustering, DBSCANClustering, HierarchicalClustering
from .quality_measures import SilhouetteScore, CalinskiHarabaszScore, DaviesBouldinScore
from .dr_techniques import PCAReduction, TSNEReduction, MDSReduction
from .dr_quality_measures import ReconstructionError, TrustworthinessScore, DistanceCorrelation

# Version info
__version__ = "3.0.0"
__author__ = "Kemal"

# Export main components for easy access
__all__ = [
    # Core components
    'Dataset',
    'ClusteringResult',
    'DRResult',
    
    # Distance measures
    'EuclideanDistance',
    'ManhattanDistance', 
    'CosineDistance',
    
    # Clustering techniques
    'KMeansClustering',
    'DBSCANClustering',
    'HierarchicalClustering',
    
    # Clustering quality measures
    'SilhouetteScore',
    'CalinskiHarabaszScore',
    'DaviesBouldinScore',
    
    # DR techniques (Project 2)
    'PCAReduction',
    'TSNEReduction',
    'MDSReduction',
    
    # DR quality measures (Project 2)
    'ReconstructionError',
    'TrustworthinessScore',
    'DistanceCorrelation',
    
    # Convenience functions
    'load_dataset',
    'get_all_distance_measures',
    'get_all_clustering_techniques', 
    'get_all_quality_measures',
    'get_all_dr_techniques',
    'get_all_dr_quality_measures'
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

def get_all_dr_techniques():
    return {
        'pca': PCAReduction,
        'tsne': TSNEReduction,
        'mds': MDSReduction
    }

def get_all_dr_quality_measures():
    return {
        'reconstruction_error': ReconstructionError,
        'trustworthiness': TrustworthinessScore,
        'distance_correlation': DistanceCorrelation
    }