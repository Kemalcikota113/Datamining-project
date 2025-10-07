"""
Dimensionality Reduction Techniques Implementation
Three different DR technique implementations using sklearn.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from .core import DimensionalityReductionTechnique, DRResult


class PCAReduction(DimensionalityReductionTechnique):
    """
    PCA (Principal Component Analysis) dimensionality reduction.
    Linear technique that finds principal components.
    """
    
    def reduce(self, dataset, distance_measure=None, **hyperparams):
        """
        Perform PCA dimensionality reduction.
        
        Args:
            dataset: Dataset object
            distance_measure: Not used by PCA
            **hyperparams: PCA hyperparameters
                - n_components: Number of components (default: 2)
                - random_state: Random state (default: 42)
        
        Returns:
            DRResult object
        """
        # Extract hyperparameters
        n_components = hyperparams.get('n_components', 2)
        random_state = hyperparams.get('random_state', 42)
        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced_data = pca.fit_transform(data_points)
        
        # Store metadata
        metadata = {
            'algorithm': 'PCA',
            'hyperparams': hyperparams,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': np.sum(pca.explained_variance_ratio_)
        }
        
        return DRResult(
            reduced_data=reduced_data,
            explained_variance=pca.explained_variance_ratio_,
            metadata=metadata
        )


class TSNEReduction(DimensionalityReductionTechnique):
    """
    t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction.
    Non-linear technique good for visualization.
    """
    
    def reduce(self, dataset, distance_measure=None, **hyperparams):
        """
        Perform t-SNE dimensionality reduction.
        
        Args:
            dataset: Dataset object
            distance_measure: DistanceMeasure object (can use custom metric)
            **hyperparams: t-SNE hyperparameters
                - n_components: Number of components (default: 2)
                - perplexity: Perplexity parameter (default: 30)
                - random_state: Random state (default: 42)
                - metric: Distance metric (default: 'euclidean')
        
        Returns:
            DRResult object
        """
        # Extract hyperparameters
        n_components = hyperparams.get('n_components', 2)
        perplexity = hyperparams.get('perplexity', 30)
        random_state = hyperparams.get('random_state', 42)
        metric = hyperparams.get('metric', 'euclidean')
        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            metric=metric
        )
        reduced_data = tsne.fit_transform(data_points)
        
        # Store metadata
        metadata = {
            'algorithm': 't-SNE',
            'hyperparams': hyperparams,
            'kl_divergence': tsne.kl_divergence_ if hasattr(tsne, 'kl_divergence_') else None
        }
        
        return DRResult(
            reduced_data=reduced_data,
            explained_variance=None,  # t-SNE doesn't provide this
            metadata=metadata
        )


class MDSReduction(DimensionalityReductionTechnique):
    """
    MDS (Multidimensional Scaling) dimensionality reduction.
    Preserves distances between points.
    """
    
    def reduce(self, dataset, distance_measure=None, **hyperparams):
        """
        Perform MDS dimensionality reduction.
        
        Args:
            dataset: Dataset object
            distance_measure: DistanceMeasure object (can use custom distance)
            **hyperparams: MDS hyperparameters
                - n_components: Number of components (default: 2)
                - random_state: Random state (default: 42)
                - metric: Use metric MDS (default: True)
        
        Returns:
            DRResult object
        """
        # Extract hyperparameters
        n_components = hyperparams.get('n_components', 2)
        random_state = hyperparams.get('random_state', 42)
        metric = hyperparams.get('metric', True)
        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Use custom distance measure if provided
        if distance_measure is not None:
            # Compute pairwise distance matrix
            n_points = len(data_points)
            distance_matrix = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = distance_measure.calculate(data_points[i], data_points[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            # Use precomputed distance matrix
            mds = MDS(
                n_components=n_components,
                random_state=random_state,
                metric=metric,
                dissimilarity='precomputed'
            )
            reduced_data = mds.fit_transform(distance_matrix)
        else:
            # Use standard sklearn implementation
            mds = MDS(
                n_components=n_components,
                random_state=random_state,
                metric=metric
            )
            reduced_data = mds.fit_transform(data_points)
        
        # Store metadata
        metadata = {
            'algorithm': 'MDS',
            'hyperparams': hyperparams,
            'stress': mds.stress_ if hasattr(mds, 'stress_') else None
        }
        
        return DRResult(
            reduced_data=reduced_data,
            explained_variance=None,  # MDS doesn't provide this
            metadata=metadata
        )
