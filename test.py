"""Health check test for datamining framework."""

import datamining_framework as dmf
from sklearn.datasets import load_iris
import numpy as np

def test_clustering():
    """Test clustering components."""
    dataset = dmf.load_dataset(load_iris().data)
    kmeans = dmf.KMeansClustering()
    result = kmeans.cluster(dataset, n_clusters=3)
    silhouette = dmf.SilhouetteScore()
    score = silhouette.evaluate(result, dataset)
    assert 0 <= score <= 1, "Clustering failed"
    print("Clustering, check")

def test_dimensionality_reduction():
    """Test DR components."""
    dataset = dmf.load_dataset(load_iris().data)
    pca = dmf.PCAReduction()
    dr_result = pca.reduce(dataset, n_components=2)
    trustworthiness = dmf.TrustworthinessScore()
    score = trustworthiness.evaluate(dr_result, dataset)
    assert 0 <= score <= 1, "DR failed"
    print("Dimensionality Reduction, check")

def test_network_analysis():
    """Test network components."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    network = dmf.load_network(edges)
    louvain = dmf.LouvainCommunityDetection()
    communities = louvain.detect_communities(network)
    pagerank = dmf.PageRankMeasure()
    pr_values = pagerank.calculate(network)
    edge_bet = dmf.EdgeBetweennessMeasure()
    eb_values = edge_bet.calculate(network)
    assert len(pr_values) == 4, "Network analysis failed"
    print("Network Analysis, check")

def test_utilities():
    """Test utility functions."""
    dataset = dmf.load_dataset(load_iris().data)
    normalized = dmf.normalize_dataset(dataset, method='standard')
    info = dmf.get_dataset_info(normalized)
    assert abs(info['mean'][0]) < 1e-10, "Normalization failed"
    print("Utilities, check")

def main():
    print("Running framework health checks...\n")
    test_clustering()
    test_dimensionality_reduction()
    test_network_analysis()
    test_utilities()
    print("\nAll health checks passed!")
    print(f"Framework version: {dmf.__version__}")

if __name__ == "__main__":
    main()
