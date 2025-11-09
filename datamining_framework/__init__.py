from .core import Dataset, ClusteringResult, DRResult, Network, CommunityResult
from .distance_measures import EuclideanDistance, ManhattanDistance, CosineDistance
from .clustering_techniques import KMeansClustering, DBSCANClustering, HierarchicalClustering
from .quality_measures import SilhouetteScore, CalinskiHarabaszScore, DaviesBouldinScore
from .dr_techniques import PCAReduction, TSNEReduction, MDSReduction
from .dr_quality_measures import ReconstructionError, TrustworthinessScore, DistanceCorrelation
from .util import load_csv, normalize_dataset, save_csv, get_dataset_info, split_features_labels
from .community_detection import LouvainCommunityDetection, GirvanNewmanCommunityDetection, FastNewmanCommunityDetection
from .node_measures import PageRankMeasure, DegreeCentralityMeasure, BetweennessCentralityMeasure, ClosenessCentralityMeasure
from .edge_measures import EdgeBetweennessMeasure, EdgeCurrentFlowMeasure, EdgeLoadMeasure, EdgeWeightMeasure

# Version info
__version__ = "4.0.0"
__author__ = "Kemal"

# Export main components for easy access
__all__ = [
    # Core components
    'Dataset',
    'ClusteringResult',
    'DRResult',
    'Network',
    'CommunityResult',
    
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
    
    # Network components (Project 3)
    'LouvainCommunityDetection',
    'GirvanNewmanCommunityDetection',
    'FastNewmanCommunityDetection',
    'PageRankMeasure',
    'DegreeCentralityMeasure',
    'BetweennessCentralityMeasure',
    'ClosenessCentralityMeasure',
    'EdgeBetweennessMeasure',
    'EdgeCurrentFlowMeasure',
    'EdgeLoadMeasure',
    'EdgeWeightMeasure',
    
    # Utility functions
    'load_csv',
    'normalize_dataset',
    'save_csv',
    'get_dataset_info',
    'split_features_labels',
    
    # Convenience functions
    'load_dataset',
    'load_network',
    'get_all_distance_measures',
    'get_all_clustering_techniques', 
    'get_all_quality_measures',
    'get_all_dr_techniques',
    'get_all_dr_quality_measures',
    'get_all_community_detection_techniques',
    'get_all_node_measures',
    'get_all_edge_measures'
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

def load_network(data, directed=False):
    """Load network from data or file"""
    return Network(data, directed=directed)

def get_all_community_detection_techniques():
    return {
        'louvain': LouvainCommunityDetection,
        'girvan_newman': GirvanNewmanCommunityDetection,
        'fast_newman': FastNewmanCommunityDetection
    }

def get_all_node_measures():
    return {
        'pagerank': PageRankMeasure,
        'degree_centrality': DegreeCentralityMeasure,
        'betweenness_centrality': BetweennessCentralityMeasure,
        'closeness_centrality': ClosenessCentralityMeasure
    }

def get_all_edge_measures():
    return {
        'edge_betweenness': EdgeBetweennessMeasure,
        'edge_current_flow': EdgeCurrentFlowMeasure,
        'edge_load': EdgeLoadMeasure,
        'edge_weight': EdgeWeightMeasure
    }