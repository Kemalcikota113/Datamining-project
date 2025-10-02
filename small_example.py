"""
Small Example - Data Mining Framework Basic Usage
Simple demonstration of core framework features.
"""

import datamining_framework as dmf
from sklearn.datasets import load_iris # if we want to use iris dataset

def main():
    print("Data Mining Framework - Quick Example")
    print("=" * 45)

    # load data/artificial_dataset.csv
    #dataset = dmf.load_dataset("data/artificial_dataset.csv")
    #print(f"   Dataset shape: {dataset.get_shape()}")
    #print(f"   Features: {len(dataset.get_features())}")

    # 1. Load dataset
    print("\n1. Loading dataset...")
    dataset = dmf.load_dataset("data/artificial_dataset.csv")
    print(f"   Dataset shape: {dataset.get_shape()}")
    print(f"   Features: {len(dataset.get_features())}")
    
    
    # 2. Create distance measure
    print("\n2. Using distance measure...")
    euclidean = dmf.EuclideanDistance()
    point1 = dataset.get_data_points()[0]
    point2 = dataset.get_data_points()[1]
    distance = euclidean.calculate(point1, point2)
    print(f"   Distance between first two points: {distance:.3f}")
    
    # 3. Apply clustering
    print("\n3. Applying clustering...")
    kmeans = dmf.KMeansClustering()
    result = kmeans.cluster(dataset, n_clusters=3)
    print(f"   Found {result.n_clusters} clusters")
    print(f"   Cluster labels: {result.get_labels()[:10]}...")  # First 10 labels
    
    # 4. Evaluate quality
    print("\n4. Evaluating clustering quality...")
    silhouette = dmf.SilhouetteScore()
    score = silhouette.evaluate(result, dataset)
    print(f"   Silhouette score: {score:.3f}")
    
    # 5. Try different algorithms
    print("\n5. Comparing algorithms...")
    algorithms = {
        'K-Means': dmf.KMeansClustering(),
        'DBSCAN': dmf.DBSCANClustering(),
        'Hierarchical': dmf.HierarchicalClustering()
    }
    
    for name, algorithm in algorithms.items():
        result = algorithm.cluster(dataset, n_clusters=3, eps=0.5, min_samples=5)
        score = silhouette.evaluate(result, dataset)
        print(f"   {name}: {result.n_clusters} clusters, score: {score:.3f}")
    
    print("\n" + "=" * 45)
    print("✅ Framework demonstration complete!")
    print("\nNext steps:")
    print("- Run full pipeline: python main.py")
    print("- See detailed results: python project1_pipeline.py")

if __name__ == "__main__":
    main()