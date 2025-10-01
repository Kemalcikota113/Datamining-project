"""
Data Mining Framework - Main Demo
Simplified framework focused on Project 1 requirements.
"""

import datamining_framework as dmf
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("DATA MINING FRAMEWORK - PROJECT 1 DEMO")
    print("=" * 60)
    
    # Load sample data
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    print(f"\n1. Loading Iris dataset...")
    print(f"   Shape: {iris_data.shape}")
    
    # Create dataset component
    dataset = dmf.load_dataset(iris_data)
    print(f"   Dataset features: {len(dataset.get_features())}")
    print(f"   Data points: {len(dataset.get_data_points())}")
    
    # Test distance measures
    print(f"\n2. Testing distance measures...")
    euclidean = dmf.EuclideanDistance()
    manhattan = dmf.ManhattanDistance()
    cosine = dmf.CosineDistance()
    
    point1 = dataset.get_data_points()[0]
    point2 = dataset.get_data_points()[1]
    
    print(f"   Euclidean distance: {euclidean.calculate(point1, point2):.4f}")
    print(f"   Manhattan distance: {manhattan.calculate(point1, point2):.4f}")
    print(f"   Cosine distance: {cosine.calculate(point1, point2):.4f}")
    
    # Test clustering techniques
    print(f"\n3. Testing clustering techniques...")
    
    # K-Means
    kmeans = dmf.KMeansClustering()
    kmeans_result = kmeans.cluster(dataset, distance_measure=euclidean, n_clusters=3)
    print(f"   K-Means: {kmeans_result.n_clusters} clusters found")
    
    # DBSCAN
    dbscan = dmf.DBSCANClustering()
    dbscan_result = dbscan.cluster(dataset, distance_measure=euclidean, eps=0.5, min_samples=5)
    print(f"   DBSCAN: {dbscan_result.n_clusters} clusters found")
    
    # Hierarchical
    hierarchical = dmf.HierarchicalClustering()
    hier_result = hierarchical.cluster(dataset, distance_measure=euclidean, n_clusters=3)
    print(f"   Hierarchical: {hier_result.n_clusters} clusters found")
    
    # Test quality measures
    print(f"\n4. Testing quality measures...")
    
    silhouette = dmf.SilhouetteScore()
    calinski = dmf.CalinskiHarabaszScore()
    davies_bouldin = dmf.DaviesBouldinScore()
    
    # Evaluate K-Means results
    print(f"   K-Means quality:")
    print(f"     Silhouette: {silhouette.evaluate(kmeans_result, dataset):.4f}")
    print(f"     Calinski-Harabasz: {calinski.evaluate(kmeans_result, dataset):.4f}")
    print(f"     Davies-Bouldin: {davies_bouldin.evaluate(kmeans_result, dataset):.4f}")
    
    # Framework components summary
    print(f"\n5. Framework components available:")
    print(f"   Distance measures: {list(dmf.get_all_distance_measures().keys())}")
    print(f"   Clustering techniques: {list(dmf.get_all_clustering_techniques().keys())}")
    print(f"   Quality measures: {list(dmf.get_all_quality_measures().keys())}")
    
    print(f"\n6. Framework design:")
    print(f"   ✓ Modular component-based architecture")
    print(f"   ✓ Abstract base classes for extensibility")
    print(f"   ✓ Ready for Project 2 (Dimensionality Reduction)")
    print(f"   ✓ Clean API following SOLID principles")
    print(f"   ✓ Supports custom hyperparameters")
    
    print("\n" + "=" * 60)
    print("🎉 FRAMEWORK READY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run full pipeline: python project1_pipeline.py")
    print("2. Test 27 combinations as required")
    print("3. Ready to add Project 2 components")

if __name__ == "__main__":
    main()
