import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from .core import ClusteringTechnique, ClusteringResult


class KMeansClustering(ClusteringTechnique):

    # K-Means clustering implementation.
    
    def cluster(self, dataset, distance_measure=None, **hyperparams):

        # Extract hyperparameters with defaults
        n_clusters = hyperparams.get('n_clusters', 3)
        random_state = hyperparams.get('random_state', 42)
        max_iter = hyperparams.get('max_iter', 300)
        init = hyperparams.get('init', 'k-means++')
        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Create and fit K-Means model
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            init=init,
            n_init=10
        )
        
        labels = kmeans.fit_predict(data_points)
        centers = kmeans.cluster_centers_
        
        # Store metadata
        metadata = {
            'algorithm': 'K-Means',
            'hyperparams': hyperparams,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
        
        return ClusteringResult(labels, centers, metadata)


class DBSCANClustering(ClusteringTechnique):
    
    # DBSCAN clustering implementation.

    
    def cluster(self, dataset, distance_measure=None, **hyperparams):

        # Extract hyperparameters with defaults
        eps = hyperparams.get('eps', 0.5)
        min_samples = hyperparams.get('min_samples', 5)
        metric = hyperparams.get('metric', 'euclidean')
        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Use custom distance measure if provided
        if distance_measure is not None:
            # Compute pairwise distance matrix using custom measure
            n_points = len(data_points)
            distance_matrix = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = distance_measure.calculate(data_points[i], data_points[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            # Use precomputed distance matrix
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = dbscan.fit_predict(distance_matrix)
        else:
            # Use standard sklearn implementation
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = dbscan.fit_predict(data_points)
        
        # DBSCAN doesn't have explicit cluster centers, compute them
        centers = None
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 or (len(unique_labels) == 1 and unique_labels[0] != -1):
            centers = []
            for label in unique_labels:
                if label != -1:  # Exclude noise points
                    cluster_points = data_points[labels == label]
                    center = np.mean(cluster_points, axis=0)
                    centers.append(center)
            centers = np.array(centers) if centers else None
        
        # Store metadata
        metadata = {
            'algorithm': 'DBSCAN',
            'hyperparams': hyperparams,
            'n_noise_points': np.sum(labels == -1),
            'core_sample_indices': dbscan.core_sample_indices_ if hasattr(dbscan, 'core_sample_indices_') else None
        }
        
        return ClusteringResult(labels, centers, metadata)


class HierarchicalClustering(ClusteringTechnique):
    
    # Hierarchical clustering implementation.
 
    
    def cluster(self, dataset, distance_measure=None, **hyperparams):

        # Extract hyperparameters with defaults
        n_clusters = hyperparams.get('n_clusters', 3)
        linkage = hyperparams.get('linkage', 'ward')
        affinity = hyperparams.get('affinity', 'euclidean')
        
        # Get data points
        data_points = dataset.get_data_points()
        
        # Use custom distance measure if provided
        if distance_measure is not None:
            # Compute pairwise distance matrix using custom measure
            n_points = len(data_points)
            distance_matrix = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = distance_measure.calculate(data_points[i], data_points[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            # Use precomputed distance matrix
            # Note: ward linkage requires euclidean affinity
            if linkage == 'ward':
                linkage = 'complete'  # Change to compatible linkage
            
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric='precomputed'
            )
            labels = hierarchical.fit_predict(distance_matrix)
        else:
            # Use standard sklearn implementation
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=affinity
            )
            labels = hierarchical.fit_predict(data_points)
        
        # Compute cluster centers
        centers = []
        for label in range(n_clusters):
            cluster_points = data_points[labels == label]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                centers.append(center)
        centers = np.array(centers)
        
        # Store metadata
        metadata = {
            'algorithm': 'Hierarchical',
            'hyperparams': hyperparams,
            'n_leaves': hierarchical.n_leaves_ if hasattr(hierarchical, 'n_leaves_') else None,
            'n_connected_components': hierarchical.n_connected_components_ if hasattr(hierarchical, 'n_connected_components_') else None
        }
        
        return ClusteringResult(labels, centers, metadata)
