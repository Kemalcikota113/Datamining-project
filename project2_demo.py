"""
Project 2 - Dimensionality Reduction Demo
Demonstrates DR techniques and DR + Clustering pipeline.
"""

import datamining_framework as dmf
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
import time

def test_dr_techniques():
    """Test dimensionality reduction techniques"""
    print("=" * 70)
    print("PROJECT 2: DIMENSIONALITY REDUCTION DEMO")
    print("=" * 70)
    
    # Load dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    dataset = dmf.load_dataset(iris.data)
    
    print(f"\n1. Original Dataset:")
    print(f"Shape: {dataset.get_shape()}")
    print(f"Dimensions: {dataset.get_shape()[1]}")
    
    # Test DR techniques
    print(f"\n2. Testing Dimensionality Reduction Techniques:")
    
    # PCA
    print(f"\n   a) PCA Reduction:")
    pca = dmf.PCAReduction()
    pca_result = pca.reduce(dataset, n_components=2)
    print(f"Reduced shape: {pca_result.get_reduced_data().shape}")
    print(f"Variance explained: {pca_result.metadata['total_variance_explained']:.3f}")
    
    # t-SNE
    print(f"\n   b) t-SNE Reduction:")
    tsne = dmf.TSNEReduction()
    tsne_result = tsne.reduce(dataset, n_components=2, perplexity=30)
    print(f"Reduced shape: {tsne_result.get_reduced_data().shape}")
    
    # MDS
    print(f"\n   c) MDS Reduction:")
    mds = dmf.MDSReduction()
    mds_result = mds.reduce(dataset, n_components=2)
    print(f"Reduced shape: {mds_result.get_reduced_data().shape}")
    
    # Test quality measures
    print(f"\n3. Testing DR Quality Measures:")
    
    reconstruction = dmf.ReconstructionError()
    trustworthiness = dmf.TrustworthinessScore()
    distance_corr = dmf.DistanceCorrelation()
    
    print(f"\n   PCA Quality:")
    print(f"Reconstruction Error: {reconstruction.evaluate(pca_result, dataset):.4f}")
    print(f"Trustworthiness: {trustworthiness.evaluate(pca_result, dataset):.4f}")
    print(f"Distance Correlation: {distance_corr.evaluate(pca_result, dataset):.4f}")
    
    return dataset, pca_result, tsne_result


def test_dr_then_clustering():
    """Test DR + Clustering pipeline"""
    print("\n" + "=" * 70)
    print("DR + CLUSTERING PIPELINE COMPARISON")
    print("=" * 70)
    
    # Load Wine dataset (higher dimensional)
    wine = load_wine()
    original_dataset = dmf.load_dataset(wine.data)
    
    print(f"\nDataset: Wine")
    print(f"Original dimensions: {original_dataset.get_shape()[1]}")
    
    # Test 1: Clustering on original data
    print(f"\n1. Clustering on ORIGINAL high-dimensional data:")
    kmeans = dmf.KMeansClustering()
    
    start_time = time.time()
    original_result = kmeans.cluster(original_dataset, n_clusters=3)
    original_time = time.time() - start_time
    
    silhouette = dmf.SilhouetteScore()
    original_score = silhouette.evaluate(original_result, original_dataset)
    
    print(f"Time: {original_time:.4f}s")
    print(f"Clusters found: {original_result.n_clusters}")
    print(f"Silhouette score: {original_score:.4f}")
    
    # Test 2: DR then clustering
    print(f"\n2. Clustering on REDUCED data (PCA):")
    
    # Reduce dimensions
    pca = dmf.PCAReduction()
    reduced_result = pca.reduce(original_dataset, n_components=5)
    reduced_dataset = reduced_result.get_reduced_dataset()
    
    print(f"Reduced to {reduced_dataset.get_shape()[1]} dimensions")
    print(f"Variance retained: {reduced_result.metadata['total_variance_explained']:.3f}")
    
    # Cluster reduced data
    start_time = time.time()
    reduced_cluster_result = kmeans.cluster(reduced_dataset, n_clusters=3)
    reduced_time = time.time() - start_time
    
    reduced_score = silhouette.evaluate(reduced_cluster_result, reduced_dataset)
    
    print(f"Time: {reduced_time:.4f}s")
    print(f"Clusters found: {reduced_cluster_result.n_clusters}")
    print(f"Silhouette score: {reduced_score:.4f}")
    
    # Comparison
    print(f"\n3. Comparison:")
    print(f"   Speed improvement: {original_time/reduced_time:.2f}x faster")
    print(f"   Quality change: {reduced_score - original_score:+.4f}")
    
    if reduced_score >= original_score * 0.95:  # Within 5%
        print(f"   ✅ DR maintains quality while improving speed!")
    else:
        print(f"   ⚠️  DR trades some quality for speed")


def comprehensive_test():
    """Test multiple DR techniques with clustering and export results to CSV"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DR + CLUSTERING TEST")
    print("=" * 70)
    
    iris = load_iris()
    dataset = dmf.load_dataset(iris.data)
    
    dr_techniques = {
        'PCA': dmf.PCAReduction(),
        't-SNE': dmf.TSNEReduction(),
        'MDS': dmf.MDSReduction()
    }
    
    clustering_techniques = {
        'K-Means': dmf.KMeansClustering(),
        'Hierarchical': dmf.HierarchicalClustering()
    }
    
    dr_quality_measures = {
        'Reconstruction Error': dmf.ReconstructionError(),
        'Trustworthiness': dmf.TrustworthinessScore(),
        'Distance Correlation': dmf.DistanceCorrelation()
    }
    
    clustering_quality_measures = {
        'Silhouette': dmf.SilhouetteScore(),
        'Calinski-Harabasz': dmf.CalinskiHarabaszScore(),
        'Davies-Bouldin': dmf.DaviesBouldinScore()
    }
    
    print(f"\nTesting {len(dr_techniques)} DR × {len(clustering_techniques)} Clustering × {len(clustering_quality_measures)} Quality")
    print(f"Total combinations: {len(dr_techniques) * len(clustering_techniques) * len(clustering_quality_measures)}\n")
    
    # Store all results
    all_results = []
    run_count = 0
    
    for dr_name, dr_technique in dr_techniques.items():
        # Reduce dimensions
        dr_result = dr_technique.reduce(dataset, n_components=2)
        reduced_dataset = dr_result.get_reduced_dataset()
        
        # Evaluate DR quality
        dr_quality_scores = {}
        for dr_qual_name, dr_qual_measure in dr_quality_measures.items():
            dr_quality_scores[dr_qual_name] = dr_qual_measure.evaluate(dr_result, dataset)
        
        print(f"{dr_name} Reduction:")
        print(f"  DR Quality - Reconstruction: {dr_quality_scores['Reconstruction Error']:.4f}, "
              f"Trustworthiness: {dr_quality_scores['Trustworthiness']:.4f}, "
              f"Distance Corr: {dr_quality_scores['Distance Correlation']:.4f}")
        
        for cluster_name, cluster_technique in clustering_techniques.items():
            # Cluster reduced data
            cluster_result = cluster_technique.cluster(reduced_dataset, n_clusters=3)
            
            # Evaluate with all clustering quality measures
            for clust_qual_name, clust_qual_measure in clustering_quality_measures.items():
                run_count += 1
                score = clust_qual_measure.evaluate(cluster_result, reduced_dataset)
                
                # Store result
                result = {
                    'run': run_count,
                    'dr_technique': dr_name,
                    'n_components': 2,
                    'clustering_technique': cluster_name,
                    'quality_measure': clust_qual_name,
                    'clustering_score': score,
                    'n_clusters': cluster_result.n_clusters,
                    'dr_reconstruction_error': dr_quality_scores['Reconstruction Error'],
                    'dr_trustworthiness': dr_quality_scores['Trustworthiness'],
                    'dr_distance_correlation': dr_quality_scores['Distance Correlation']
                }
                all_results.append(result)
            
            # Print summary for this combination
            silhouette_score = clustering_quality_measures['Silhouette'].evaluate(cluster_result, reduced_dataset)
            print(f"  + {cluster_name}: {cluster_result.n_clusters} clusters, Silhouette: {silhouette_score:.4f}")
        print()
    
    # Export to CSV
    results_df = pd.DataFrame(all_results)
    csv_filename = 'project2_results.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Detailed results exported to: {csv_filename}\n")
    
    # Print summary statistics
    print("SUMMARY STATISTICS:")
    print("-" * 70)
    
    # Best results by clustering quality measure
    for qual_measure in clustering_quality_measures.keys():
        measure_results = results_df[results_df['quality_measure'] == qual_measure]
        if qual_measure == 'Davies-Bouldin':
            best_idx = measure_results['clustering_score'].idxmin()  # Lower is better
        else:
            best_idx = measure_results['clustering_score'].idxmax()  # Higher is better
        
        if pd.notna(best_idx):
            best = measure_results.loc[best_idx]
            print(f"\nBest {qual_measure}:")
            print(f"  Score: {best['clustering_score']:.4f}")
            print(f"  DR: {best['dr_technique']}, Clustering: {best['clustering_technique']}")
    
    return all_results


def main():
    # Test DR techniques
    test_dr_techniques()
    
    # Test DR + Clustering pipeline
    test_dr_then_clustering()
    
    # Comprehensive test
    comprehensive_test()
    
    print("=" * 70)
    print("PROJECT 2 DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. DR reduces dimensions while preserving data structure")
    print("2. Clustering on reduced data can be faster")
    print("3. Quality may improve or slightly decrease depending on data")
    print("4. All components work together seamlessly!")


if __name__ == "__main__":
    main()
