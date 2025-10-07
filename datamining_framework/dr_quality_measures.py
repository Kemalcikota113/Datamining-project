"""
Dimensionality Reduction Quality Measures Implementation
Three different DR quality measure implementations.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from .core import DRQualityMeasure


class ReconstructionError(DRQualityMeasure):
    """
    Reconstruction error quality measure.
    Measures how well the reduced data can be reconstructed.
    Range: [0, +inf), lower is better.
    """
    
    def evaluate(self, dr_result, original_dataset):
        """
        Calculate reconstruction error.
        
        Args:
            dr_result: DRResult object
            original_dataset: Original Dataset object
            
        Returns:
            float: Reconstruction error (lower is better)
        """
        original_data = original_dataset.get_data_points()
        reduced_data = dr_result.get_reduced_data()
        
        # For PCA, we can compute exact reconstruction error
        # For other methods, use approximation based on variance
        if 'PCA' in dr_result.metadata.get('algorithm', ''):
            # Use explained variance ratio
            variance_ratio = dr_result.metadata.get('total_variance_explained', 0)
            # Reconstruction error = 1 - variance explained
            return 1.0 - variance_ratio
        else:
            # Approximate reconstruction error using variance preservation
            original_var = np.var(original_data)
            reduced_var = np.var(reduced_data)
            
            if original_var == 0:
                return 0.0
            
            return 1.0 - (reduced_var / original_var)


class TrustworthinessScore(DRQualityMeasure):
    """
    Trustworthiness score quality measure.
    Measures how well local neighborhoods are preserved.
    Range: [0, 1], higher is better.
    """
    
    def evaluate(self, dr_result, original_dataset):
        """
        Calculate trustworthiness score.
        
        Args:
            dr_result: DRResult object
            original_dataset: Original Dataset object
            
        Returns:
            float: Trustworthiness score (higher is better)
        """
        from sklearn.manifold import trustworthiness
        
        original_data = original_dataset.get_data_points()
        reduced_data = dr_result.get_reduced_data()
        
        # Use k = min(12, n_samples - 1) as recommended
        n_samples = len(original_data)
        k = min(12, n_samples - 1)
        
        if k < 1:
            return 0.0
        
        return trustworthiness(original_data, reduced_data, n_neighbors=k)


class DistanceCorrelation(DRQualityMeasure):
    """
    Distance correlation quality measure.
    Measures correlation between original and reduced distance matrices.
    Range: [-1, 1], higher is better.
    """
    
    def evaluate(self, dr_result, original_dataset):
        """
        Calculate distance correlation using Spearman's rank correlation.
        
        Args:
            dr_result: DRResult object
            original_dataset: Original Dataset object
            
        Returns:
            float: Distance correlation (higher is better)
        """
        original_data = original_dataset.get_data_points()
        reduced_data = dr_result.get_reduced_data()
        
        # Compute pairwise distances
        original_distances = pairwise_distances(original_data).flatten()
        reduced_distances = pairwise_distances(reduced_data).flatten()
        
        # Calculate Spearman correlation
        correlation, _ = spearmanr(original_distances, reduced_distances)
        
        # Handle NaN values
        if np.isnan(correlation):
            return 0.0
        
        return correlation
