"""
Data Mining Framework - Core Components
Modular design following the project requirements structure.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Dataset:
    """
    Component for handling structured data.
    Requirement 1: Dataset component with methods to get data points and features.
    """
    
    def __init__(self, data):
        """
        Initialize dataset.
        
        Args:
            data: pandas DataFrame, numpy array, or file path
        """
        if isinstance(data, str):
            # Load from file
            if data.endswith('.csv'):
                self.data = pd.read_csv(data)
            elif data.endswith('.xlsx') or data.endswith('.xls'):
                self.data = pd.read_excel(data)
            else:
                raise ValueError("Unsupported file format")
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Unsupported data type")
    
    def get_data_points(self):
        """
        Get data points (rows).
        
        Returns:
            numpy array of data points
        """
        return self.data.values
    
    def get_features(self):
        """
        Get features (columns).
        
        Returns:
            list of feature names or indices
        """
        return self.data.columns.tolist()
    
    def get_shape(self):
        """Get dataset shape"""
        return self.data.shape
    
    def get_dataframe(self):
        """Get underlying DataFrame"""
        return self.data.copy()


class DistanceMeasure(ABC):
    """
    Abstract base class for distance measures.
    Requirement 2: Distance measure component that takes two points and returns distance.
    """
    
    @abstractmethod
    def calculate(self, point1, point2):
        """
        Calculate distance between two points.
        
        Args:
            point1: First data point (array-like)
            point2: Second data point (array-like)
            
        Returns:
            float: Distance value
        """
        pass


class ClusteringTechnique(ABC):
    """
    Abstract base class for clustering techniques.
    Requirement 3: Clustering component that accepts dataset and hyperparameters.
    """
    
    @abstractmethod
    def cluster(self, dataset, distance_measure=None, **hyperparams):
        """
        Perform clustering on dataset.
        
        Args:
            dataset: Dataset object
            distance_measure: DistanceMeasure object (optional)
            **hyperparams: Algorithm-specific hyperparameters
            
        Returns:
            ClusteringResult object
        """
        pass


class QualityMeasure(ABC):
    """
    Abstract base class for quality measures.
    Requirement 4: Quality measure for clustering that takes clustering results and dataset.
    """
    
    @abstractmethod
    def evaluate(self, clustering_result, dataset):
        """
        Evaluate clustering quality.
        
        Args:
            clustering_result: ClusteringResult object
            dataset: Original Dataset object
            
        Returns:
            float: Quality score
        """
        pass


class ClusteringResult:
    """
    Container for clustering results.
    """
    
    def __init__(self, labels, centers=None, metadata=None):
        """
        Initialize clustering result.
        
        Args:
            labels: Cluster labels for each data point
            centers: Cluster centers (optional)
            metadata: Additional information (optional)
        """
        self.labels = np.array(labels)
        self.centers = centers
        self.metadata = metadata or {}
        self.n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise points (-1)
    
    def get_labels(self):
        """Get cluster labels"""
        return self.labels
    
    def get_centers(self):
        """Get cluster centers"""
        return self.centers
    
    def get_clusters(self):
        """
        Get clusters as list of lists of point indices.
        
        Returns:
            list: List of clusters, each cluster is a list of point indices
        """
        clusters = []
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            if label >= 0:  # Exclude noise points (-1)
                cluster_points = np.where(self.labels == label)[0].tolist()
                clusters.append(cluster_points)
        
        return clusters


# Extensible base classes for future components (Project 2 and beyond)

class DimensionalityReductionTechnique(ABC):
    """
    Abstract base class for dimensionality reduction techniques.
    For Project 2: DR component that accepts dataset and hyperparameters.
    """
    
    @abstractmethod
    def reduce(self, dataset, distance_measure=None, **hyperparams):
        """
        Perform dimensionality reduction on dataset.
        
        Args:
            dataset: Dataset object
            distance_measure: DistanceMeasure object (optional)
            **hyperparams: Algorithm-specific hyperparameters
            
        Returns:
            DRResult object with reduced dataset
        """
        pass


class DRQualityMeasure(ABC):
    """
    Abstract base class for dimensionality reduction quality measures.
    For Project 2: Quality measure for DR results.
    """
    
    @abstractmethod
    def evaluate(self, original_dataset, reduced_dataset):
        """
        Evaluate dimensionality reduction quality.
        
        Args:
            original_dataset: Original Dataset object
            reduced_dataset: Reduced Dataset object
            
        Returns:
            float: Quality score
        """
        pass


class DRResult:
    """
    Container for dimensionality reduction results.
    For Project 2: Results of DR techniques.
    """
    
    def __init__(self, reduced_data, explained_variance=None, metadata=None):
        """
        Initialize DR result.
        
        Args:
            reduced_data: Reduced data array
            explained_variance: Variance explained (optional)
            metadata: Additional information (optional)
        """
        self.reduced_data = reduced_data
        self.explained_variance = explained_variance
        self.metadata = metadata or {}
    
    def get_reduced_dataset(self):
        """Get reduced data as Dataset object"""
        return Dataset(self.reduced_data)
    
    def get_reduced_data(self):
        """Get reduced data as numpy array"""
        return self.reduced_data
