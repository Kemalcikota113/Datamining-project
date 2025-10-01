"""
Project 1 Requirements Test Pipeline
Tests 3 clustering techniques × 3 datasets × 3 quality measures = 27 runs
"""

import pandas as pd
from sklearn.datasets import load_iris, load_wine, make_blobs
import datamining_framework as dmf

def create_test_datasets():
    """
    Create three different datasets for testing.
    
    Returns:
        list: List of (dataset_name, Dataset object) tuples
    """
    datasets = []
    
    # Dataset 1: Iris dataset
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    datasets.append(("Iris", dmf.load_dataset(iris_data)))
    
    # Dataset 2: Wine dataset
    wine = load_wine()
    wine_data = pd.DataFrame(wine.data, columns=wine.feature_names)
    datasets.append(("Wine", dmf.load_dataset(wine_data)))
    
    # Dataset 3: Synthetic blob dataset
    blob_X, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                          random_state=42, cluster_std=1.5)
    blob_data = pd.DataFrame(blob_X, columns=['feature1', 'feature2'])
    datasets.append(("Synthetic_Blobs", dmf.load_dataset(blob_data)))
    
    return datasets

def run_project1_pipeline():
    """
    Run the complete Project 1 pipeline with all required combinations.
    3 clustering techniques × 3 datasets × 3 quality measures = 27 runs
    """
    print("=" * 80)
    print("PROJECT 1 REQUIREMENTS TEST PIPELINE")
    print("=" * 80)
    
    # Create test datasets
    datasets = create_test_datasets()
    
    # Get all components
    distance_measures = {
        'euclidean': dmf.EuclideanDistance(),
        'manhattan': dmf.ManhattanDistance(),
        'cosine': dmf.CosineDistance()
    }
    
    clustering_techniques = {
        'K-Means': dmf.KMeansClustering(),
        'DBSCAN': dmf.DBSCANClustering(),
        'Hierarchical': dmf.HierarchicalClustering()
    }
    
    quality_measures = {
        'Silhouette': dmf.SilhouetteScore(),
        'Calinski-Harabasz': dmf.CalinskiHarabaszScore(),
        'Davies-Bouldin': dmf.DaviesBouldinScore()
    }
    
    # Store all results for final summary
    all_results = []
    run_count = 0
    
    print(f"\nTesting {len(clustering_techniques)} techniques × {len(datasets)} datasets × {len(quality_measures)} quality measures")
    print(f"Total runs: {len(clustering_techniques) * len(datasets) * len(quality_measures)}\n")
    
    # Run all combinations
    for dataset_name, dataset in datasets:
        print(f"\n{'='*20} DATASET: {dataset_name} {'='*20}")
        print(f"Dataset shape: {dataset.get_shape()}")
        
        for tech_name, technique in clustering_techniques.items():
            print(f"\n--- Clustering Technique: {tech_name} ---")
            
            # Set appropriate hyperparameters for each technique and dataset
            if tech_name == 'K-Means':
                hyperparams = {'n_clusters': 3, 'random_state': 42}
                distance_measure = distance_measures['euclidean']
            elif tech_name == 'DBSCAN':
                # Adjust eps based on dataset
                if dataset_name == 'Synthetic_Blobs':
                    hyperparams = {'eps': 1.0, 'min_samples': 5}
                else:
                    hyperparams = {'eps': 0.5, 'min_samples': 5}
                distance_measure = distance_measures['euclidean']
            else:  # Hierarchical
                hyperparams = {'n_clusters': 3, 'linkage': 'ward'}
                distance_measure = distance_measures['euclidean']
            
            # Perform clustering
            try:
                clustering_result = technique.cluster(
                    dataset, 
                    distance_measure=distance_measure,
                    **hyperparams
                )
                
                print(f"Clustering completed. Found {clustering_result.n_clusters} clusters.")
                
                # Test all quality measures
                for quality_name, quality_measure in quality_measures.items():
                    run_count += 1
                    
                    try:
                        quality_score = quality_measure.evaluate(clustering_result, dataset)
                        
                        # Store result
                        result = {
                            'run': run_count,
                            'dataset': dataset_name,
                            'technique': tech_name,
                            'quality_measure': quality_name,
                            'score': quality_score,
                            'n_clusters': clustering_result.n_clusters,
                            'hyperparams': hyperparams
                        }
                        all_results.append(result)
                        
                        print(f"  {quality_name}: {quality_score:.4f}")
                        
                    except Exception as e:
                        print(f"  {quality_name}: ERROR - {str(e)}")
                        
            except Exception as e:
                print(f"Clustering failed: {str(e)}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL 27 RUNS")
    print("=" * 80)
    
    if all_results:
        # Create summary DataFrame
        results_df = pd.DataFrame(all_results)
        
        print(f"\nCompleted {len(all_results)} successful runs out of {len(clustering_techniques) * len(datasets) * len(quality_measures)} total runs\n")
        
        # Summary by technique
        print("RESULTS BY CLUSTERING TECHNIQUE:")
        print("-" * 40)
        for tech in clustering_techniques.keys():
            tech_results = results_df[results_df['technique'] == tech]
            if len(tech_results) > 0:
                avg_scores = tech_results.groupby('quality_measure')['score'].mean()
                print(f"\n{tech}:")
                for quality, score in avg_scores.items():
                    print(f"  Average {quality}: {score:.4f}")
        
        # Summary by dataset
        print("\nRESULTS BY DATASET:")
        print("-" * 40)
        for dataset in [d[0] for d in datasets]:
            dataset_results = results_df[results_df['dataset'] == dataset]
            if len(dataset_results) > 0:
                avg_scores = dataset_results.groupby('quality_measure')['score'].mean()
                print(f"\n{dataset}:")
                for quality, score in avg_scores.items():
                    print(f"  Average {quality}: {score:.4f}")
        
        # Best results for each quality measure
        print("\nBEST RESULTS BY QUALITY MEASURE:")
        print("-" * 40)
        for quality in quality_measures.keys():
            quality_results = results_df[results_df['quality_measure'] == quality]
            if len(quality_results) > 0:
                if quality == 'Davies-Bouldin':
                    # Lower is better for Davies-Bouldin
                    best = quality_results.loc[quality_results['score'].idxmin()]
                else:
                    # Higher is better for Silhouette and Calinski-Harabasz
                    best = quality_results.loc[quality_results['score'].idxmax()]
                
                print(f"\n{quality}:")
                print(f"  Best: {best['score']:.4f}")
                print(f"  Technique: {best['technique']}")
                print(f"  Dataset: {best['dataset']}")
        
        # Export detailed results
        results_df.to_csv('project1_results.csv', index=False)
        print(f"\nDetailed results exported to: project1_results.csv")
        
    else:
        print("No successful runs completed.")
    
    print("\n" + "=" * 80)
    print("PROJECT 1 PIPELINE COMPLETED")
    print("=" * 80)

def test_individual_components():
    """
    Test individual components as required.
    """
    print("\n" + "=" * 80)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 80)
    
    # Test Dataset component
    print("\n1. Testing Dataset Component:")
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataset = dmf.Dataset(iris_data)
    
    print(f"   Dataset shape: {dataset.get_shape()}")
    print(f"   Features: {dataset.get_features()[:3]}...")  # Show first 3 features
    print(f"   Data points shape: {dataset.get_data_points().shape}")
    
    # Test Distance Measures
    print("\n2. Testing Distance Measures:")
    point1 = [1.0, 2.0, 3.0]
    point2 = [4.0, 5.0, 6.0]
    
    euclidean = dmf.EuclideanDistance()
    manhattan = dmf.ManhattanDistance()
    cosine = dmf.CosineDistance()
    
    print(f"   Euclidean distance: {euclidean.calculate(point1, point2):.4f}")
    print(f"   Manhattan distance: {manhattan.calculate(point1, point2):.4f}")
    print(f"   Cosine distance: {cosine.calculate(point1, point2):.4f}")
    
    # Test Clustering Techniques
    print("\n3. Testing Clustering Techniques:")
    kmeans = dmf.KMeansClustering()
    result = kmeans.cluster(dataset, n_clusters=3)
    print(f"   K-Means found {result.n_clusters} clusters")
    print(f"   Cluster labels shape: {result.get_labels().shape}")
    print(f"   Cluster centers shape: {result.get_centers().shape if result.get_centers() is not None else 'None'}")
    
    # Test Quality Measures
    print("\n4. Testing Quality Measures:")
    silhouette = dmf.SilhouetteScore()
    score = silhouette.evaluate(result, dataset)
    print(f"   Silhouette score: {score:.4f}")

if __name__ == "__main__":
    # Test individual components first
    test_individual_components()
    
    # Run the full pipeline
    run_project1_pipeline()
