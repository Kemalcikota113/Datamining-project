# Datamining pipeline project

## Overview

A modular framework for clustering, dimensionality reduction, and network analysis. Supports multiple algorithms for each task with quality evaluation metrics.

### Project 1: Clustering
Built a clustering pipeline with distance measures (Euclidean, Manhattan, Cosine), clustering algorithms (K-Means, DBSCAN, Hierarchical), and quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin). Structured as reusable classes with a common interface for easy comparison of techniques.

### Project 2: Dimensionality Reduction
Extended the framework with DR techniques (PCA, t-SNE, MDS) and quality measures (Reconstruction Error, Trustworthiness, Distance Correlation). Designed to work seamlessly with the existing pipeline for visualizing high-dimensional data.

### Project 3: Network Analysis
Added graph analysis capabilities including community detection (Louvain, Girvan-Newman, Fast Newman), node centrality measures (PageRank, Degree, Betweenness, Closeness), and edge measures (Betweenness, Current Flow, Load). Built on NetworkX for efficient graph operations.

## How to run (For Linux and Mac)

1. go to root directory here
```
cd path/datamining-project
```
2. create venv and activate it
```
Python3 -m venv venv
source venv/bin/activate
```
3. install dependencies
```
pip install -r requirements.txt
```
4. run main
```
python3 main.py
```

5. check results
```
cat project1_results.csv
```

## How to run (Windows 11)

1. Open powershell or cmd

2. go to root directory here
```
cd path/datamining-project
```
3. create venv and activate it
```
Python -m venv venv
venv/bin/activate.ps1
```
4. install dependencies
```
pip install -r requirements.txt
```
5. run main
```
python main.py
```


## Usage

In order to get a small tutorial on how to run it, look at **`project1_demo.py`** and other demo files and play around with it.