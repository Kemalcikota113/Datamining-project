"""
Dimensionality Reduction Techniques Implementation
Three different DR technique implementations using sklearn.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from .core import DimensionalityReductionTechnique, DRResult


class PCAReduction(DimensionalityReductionTechnique):
    """
    PCA (Principal Component Analysis) dimensionality reduction.
    Linear technique that finds principal components.
    """
    
    def reduce(self, dataset, distance_measure=None, **hyperparams):

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

        # Extract hyperparameters
        n_components = hyperparams.get('n_components', 2)
        perplexity = hyperparams.get('perplexity', 30)
        learning_rate = hyperparams.get('learning_rate', 'auto')
        random_state = hyperparams.get('random_state', 42)
        metric = hyperparams.get('metric', 'euclidean')
        max_iter = hyperparams.get('max_iter', 1000)
        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
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
    Preserves pairwise distances between points using sklearn's optimized implementation.
    """
    
    def reduce(self, dataset, distance_measure=None, **hyperparams):

        # Extract hyperparameters
        n_components = hyperparams.get('n_components', 2)
        metric = hyperparams.get('metric', True)
        max_iter = hyperparams.get('max_iter', 300)
        random_state = hyperparams.get('random_state', 42)
        n_init = hyperparams.get('n_init', 4)  # Explicitly set to suppress warning

        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Use custom distance measure if provided
        if distance_measure is not None:
            # Compute pairwise distance matrix using vectorized operations
            n_points = len(data_points)
            
            # For custom distance, we still need a loop but optimize where possible
            distances = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    distances.append(distance_measure.calculate(data_points[i], data_points[j]))
            
            distance_matrix = squareform(distances)
            dissimilarity = 'precomputed'
        else:
            # Let sklearn handle distance computation
            distance_matrix = data_points
            dissimilarity = 'euclidean'
        
        # Apply MDS using sklearn's optimized implementation
        mds = MDS(
            n_components=n_components,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            dissimilarity=dissimilarity,
            n_jobs=-1  # Use all available cores for speed
        )
        reduced_data = mds.fit_transform(distance_matrix)
        
        # Store metadata
        metadata = {
            'algorithm': 'MDS',
            'hyperparams': hyperparams,
            'stress': mds.stress_,
            'n_iter': mds.n_iter_
        }
        
        return DRResult(
            reduced_data=reduced_data,
            explained_variance=None,  # MDS doesn't provide explained variance
            metadata=metadata
        )