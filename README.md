# Data Mining Framework

A modular Python framework for clustering, dimensionality reduction, and network analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Main Demo
```bash
python main.py
```

### Run Health Check
```bash
python test.py
```

### Run Project-Specific Demos
```bash
python project1_demo.py  # Clustering
python project2_demo.py  # Dimensionality Reduction
python project3_demo.py  # Network Analysis
```

## Usage

```python
import datamining_framework as dmf
from sklearn.datasets import load_iris

# Load data
dataset = dmf.load_dataset(load_iris().data)

# Normalize
normalized = dmf.normalize_dataset(dataset, method='standard')

# Cluster
kmeans = dmf.KMeansClustering()
result = kmeans.cluster(normalized, n_clusters=3)

# Evaluate
silhouette = dmf.SilhouetteScore()
score = silhouette.evaluate(result, normalized)
print(f"Silhouette score: {score:.4f}")
```

## Components

### Clustering (Project 1)
- **Distance Measures:** Euclidean, Manhattan, Cosine
- **Clustering:** K-Means, DBSCAN, Hierarchical
- **Quality Measures:** Silhouette, Calinski-Harabasz, Davies-Bouldin

### Dimensionality Reduction (Project 2)
- **DR Techniques:** PCA, t-SNE, MDS
- **DR Quality:** Reconstruction Error, Trustworthiness, Distance Correlation

### Network Analysis (Project 3)
- **Community Detection:** Louvain, Girvan-Newman, Label Propagation
- **Node Measures:** PageRank, Degree, Betweenness, Closeness
- **Edge Measures:** Edge Betweenness, Current Flow, Load, Weight

## Requirements

- Python 3.8+
- scikit-learn >= 1.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- networkx >= 2.6.0
- scipy >= 1.7.0

## Project Structure

```
datamining_framework/
├── core.py                    # Base classes and data structures
├── distance_measures.py       # Distance measure implementations
├── clustering_techniques.py   # Clustering algorithms
├── quality_measures.py        # Clustering quality measures
├── dr_techniques.py          # Dimensionality reduction
├── dr_quality_measures.py    # DR quality measures
├── community_detection.py    # Community detection algorithms
├── node_measures.py          # Node centrality measures
├── edge_measures.py          # Edge centrality measures
└── util.py                   # Utility functions
```

## License

Educational project for data mining course.
