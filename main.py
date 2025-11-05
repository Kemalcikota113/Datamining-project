"""
Simple demonstration of datamining framework usage.
Shows one algorithm from each category: clustering, DR, and network analysis.
"""

import datamining_framework as dmf
import networkx as nx
import numpy as np

print("=" * 70)
print("DATA MINING FRAMEWORK - REALISTIC USAGE EXAMPLE")
print("=" * 70)

# =============================================================================
# 1. CLUSTERING
# =============================================================================
print("\n1. CLUSTERING")
print("-" * 70)

# Load and prepare data from CSV
data = np.loadtxt('data/custom_dataset.csv', delimiter=',')
dataset = dmf.load_dataset(data)
normalized = dmf.normalize_dataset(dataset, method='standard')

# Apply clustering
kmeans = dmf.KMeansClustering()
result = kmeans.cluster(normalized, n_clusters=3, random_state=42)

# Evaluate
silhouette = dmf.SilhouetteScore()
score = silhouette.evaluate(result, normalized)

print(f"Dataset: Custom Dataset ({dataset.get_shape()[0]} samples, {dataset.get_shape()[1]} features)")
print(f"Algorithm: K-Means")
print(f"Clusters found: {result.n_clusters}")
print(f"Silhouette score: {score:.4f}")

# =============================================================================
# 2. DIMENSIONALITY REDUCTION
# =============================================================================
print("\n2. DIMENSIONALITY REDUCTION")
print("-" * 70)

# Load higher-dimensional data from CSV
data = np.loadtxt('data/artificial_dataset.csv', delimiter=',')
dataset = dmf.load_dataset(data)

# Apply DR
pca = dmf.PCAReduction()
dr_result = pca.reduce(dataset, n_components=2)

# Evaluate
trustworthiness = dmf.TrustworthinessScore()
score = trustworthiness.evaluate(dr_result, dataset)

print(f"Dataset: Artificial Dataset ({dataset.get_shape()[0]} samples, {dataset.get_shape()[1]} features)")
print(f"Algorithm: PCA")
print(f"Reduced to: {dr_result.n_components} components")
print(f"Variance explained: {dr_result.metadata['total_variance_explained']:.3f}")
print(f"Trustworthiness: {score:.4f}")

# =============================================================================
# 3. NETWORK ANALYSIS
# =============================================================================
print("\n3. NETWORK ANALYSIS")
print("-" * 70)

# Load network
G = nx.karate_club_graph()
network = dmf.load_network(list(G.edges()))

# Community detection
louvain = dmf.LouvainCommunityDetection()
communities = louvain.detect_communities(network)

# Node importance
pagerank = dmf.PageRankMeasure()
pr_values = pagerank.calculate(network)
top_nodes = sorted(pr_values.items(), key=lambda x: x[1], reverse=True)[:3]

# Edge importance
edge_bet = dmf.EdgeBetweennessMeasure()
eb_values = edge_bet.calculate(network)
top_edges = sorted(eb_values.items(), key=lambda x: x[1], reverse=True)[:3]

print(f"Network: Karate Club ({network.num_nodes()} nodes, {network.num_edges()} edges)")
print(f"Algorithm: Louvain")
print(f"Communities found: {communities.n_communities}")
print(f"Modularity: {communities.metadata['modularity']:.4f}")
print(f"\nTop 3 influential nodes (PageRank):")
for node, score in top_nodes:
    print(f"  Node {node}: {score:.4f}")
print(f"\nTop 3 bridge edges (Betweenness):")
for edge, score in top_edges:
    print(f"  {edge}: {score:.4f}")

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
