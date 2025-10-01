"""
Quality Measures Implementation
Three different quality measure implementations as required.
"""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .core import QualityMeasure


class SilhouetteScore(QualityMeasure):
    """
    Silhouette score quality measure.
    Measures how similar an object is to its own cluster compared to other clusters.
    Range: [-1, 1], higher is better.
    """
    
    def evaluate(self, clustering_result, dataset):
        """
        Calculate silhouette score for clustering result.
        
        Args:
            clustering_result: ClusteringResult object
            dataset: Original Dataset object
            
        Returns:
            float: Silhouette score (-1 to 1, higher is better)
        """
        data_points = dataset.get_data_points()
        labels = clustering_result.get_labels()
        
        # Filter out noise points (label -1) for DBSCAN
        valid_mask = labels >= 0
        if np.sum(valid_mask) < 2:
            return -1.0  # Not enough valid points for silhouette score
        
        valid_data = data_points[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Need at least 2 clusters for silhouette score
        if len(np.unique(valid_labels)) < 2:
            return -1.0
        
        return silhouette_score(valid_data, valid_labels)


class CalinskiHarabaszScore(QualityMeasure):
    """
    Calinski-Harabasz score quality measure.
    Also known as Variance Ratio Criterion.
    Ratio of between-cluster dispersion to within-cluster dispersion.
    Range: [0, +inf), higher is better.
    """
    
    def evaluate(self, clustering_result, dataset):
        """
        Calculate Calinski-Harabasz score for clustering result.
        
        Args:
            clustering_result: ClusteringResult object
            dataset: Original Dataset object
            
        Returns:
            float: Calinski-Harabasz score (higher is better)
        """
        data_points = dataset.get_data_points()
        labels = clustering_result.get_labels()
        
        # Filter out noise points (label -1) for DBSCAN
        valid_mask = labels >= 0
        if np.sum(valid_mask) < 2:
            return 0.0  # Not enough valid points
        
        valid_data = data_points[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Need at least 2 clusters for Calinski-Harabasz score
        if len(np.unique(valid_labels)) < 2:
            return 0.0
        
        return calinski_harabasz_score(valid_data, valid_labels)


class DaviesBouldinScore(QualityMeasure):
    """
    Davies-Bouldin score quality measure.
    Average similarity measure of each cluster with its most similar cluster.
    Range: [0, +inf), lower is better.
    """
    
    def evaluate(self, clustering_result, dataset):
        """
        Calculate Davies-Bouldin score for clustering result.
        
        Args:
            clustering_result: ClusteringResult object
            dataset: Original Dataset object
            
        Returns:
            float: Davies-Bouldin score (lower is better)
        """
        data_points = dataset.get_data_points()
        labels = clustering_result.get_labels()
        
        # Filter out noise points (label -1) for DBSCAN
        valid_mask = labels >= 0
        if np.sum(valid_mask) < 2:
            return float('inf')  # Worst possible score
        
        valid_data = data_points[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Need at least 2 clusters for Davies-Bouldin score
        if len(np.unique(valid_labels)) < 2:
            return float('inf')
        
        return davies_bouldin_score(valid_data, valid_labels)


class InertiaScore(QualityMeasure):
    """
    Inertia (Within-cluster sum of squares) quality measure.
    Sum of squared distances of samples to their closest cluster center.
    Range: [0, +inf), lower is better.
    
    Note: This is primarily useful for K-Means clustering.
    """
    
    def evaluate(self, clustering_result, dataset):
        """
        Calculate inertia score for clustering result.
        
        Args:
            clustering_result: ClusteringResult object
            dataset: Original Dataset object
            
        Returns:
            float: Inertia score (lower is better)
        """
        data_points = dataset.get_data_points()
        labels = clustering_result.get_labels()
        centers = clustering_result.get_centers()
        
        if centers is None:
            # Compute centers if not available
            unique_labels = np.unique(labels)
            centers = []
            for label in unique_labels:
                if label >= 0:  # Exclude noise points
                    cluster_points = data_points[labels == label]
                    if len(cluster_points) > 0:
                        center = np.mean(cluster_points, axis=0)
                        centers.append(center)
            
            if not centers:
                return float('inf')
            centers = np.array(centers)
        
        # Calculate inertia
        inertia = 0.0
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            if label >= 0:  # Exclude noise points
                cluster_points = data_points[labels == label]
                if len(cluster_points) > 0 and i < len(centers):
                    # Sum of squared distances to cluster center
                    distances_squared = np.sum((cluster_points - centers[i]) ** 2, axis=1)
                    inertia += np.sum(distances_squared)
        
        return inertia
