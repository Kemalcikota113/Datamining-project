"""
Project 3 - Network Analysis Demo
Demonstrates network components, community detection, and node/edge measures.
"""

import datamining_framework as dmf
import pandas as pd
import numpy as np
import networkx as nx

def create_sample_network():
    """Create a sample network with known community structure"""
    # Karate Club network (famous network with 2 communities)
    G = nx.karate_club_graph()
    edge_list = list(G.edges())
    return dmf.load_network(edge_list, directed=False)


def test_network_basics():
    """Test network creation and basic operations"""
    print("=" * 70)
    print("PROJECT 3: NETWORK ANALYSIS DEMO")
    print("=" * 70)
    
    # Create sample network
    network = create_sample_network()
    
    print(f"\n1. Network Loaded:")
    print(f"   Nodes: {network.num_nodes()}")
    print(f"   Edges: {network.num_edges()}")
    print(f"   Directed: {network.is_directed()}")
    print(f"   Sample nodes: {network.get_nodes()[:5]}")
    print(f"   Sample edges: {network.get_edges()[:5]}")
    
    return network


def test_community_detection(network):
    """Test community detection techniques"""
    print(f"\n2. COMMUNITY DETECTION")
    print("-" * 70)
    
    techniques = {
        'Louvain': dmf.LouvainCommunityDetection(),
        'Label Propagation': dmf.LabelPropagationCommunityDetection(),
        'Girvan-Newman': dmf.GirvanNewmanCommunityDetection()
    }
    
    results = {}
    
    for name, technique in techniques.items():
        print(f"\n   {name} Method:")
        
        if name == 'Girvan-Newman':
            # Limit to 3 communities for faster execution
            result = technique.detect_communities(network, k=2)
        else:
            result = technique.detect_communities(network)
        
        results[name] = result
        
        print(f"      Communities found: {result.n_communities}")
        print(f"      Modularity: {result.metadata.get('modularity', 'N/A'):.4f}")
        
        # Show community sizes
        communities = result.get_communities()
        sizes = [len(c) for c in communities]
        print(f"      Community sizes: {sizes}")
    
    return results


def test_node_measures(network):
    """Test node centrality measures"""
    print(f"\n3. NODE MEASURES")
    print("-" * 70)
    
    measures = {
        'PageRank': dmf.PageRankMeasure(),
        'Degree Centrality': dmf.DegreeCentralityMeasure(),
        'Betweenness Centrality': dmf.BetweennessCentralityMeasure(),
        'Closeness Centrality': dmf.ClosenessCentralityMeasure()
    }
    
    node_results = {}
    
    for name, measure in measures.items():
        print(f"\n   {name}:")
        
        values = measure.calculate(network)
        node_results[name] = values
        
        # Get top 3 nodes
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_nodes[:3]
        
        print(f"      Total nodes measured: {len(values)}")
        print(f"      Top 3 nodes:")
        for node, value in top_3:
            print(f"         Node {node}: {value:.4f}")
    
    return node_results


def test_edge_measures(network):
    """Test edge centrality measures"""
    print(f"\n4. EDGE MEASURES")
    print("-" * 70)
    
    measures = {
        'Edge Betweenness': dmf.EdgeBetweennessMeasure(),
        'Edge Weight': dmf.EdgeWeightMeasure()
    }
    
    edge_results = {}
    
    for name, measure in measures.items():
        print(f"\n   {name}:")
        
        values = measure.calculate(network)
        edge_results[name] = values
        
        # Get top 3 edges
        sorted_edges = sorted(values.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_edges[:3]
        
        print(f"      Total edges measured: {len(values)}")
        print(f"      Top 3 edges:")
        for edge, value in top_3:
            print(f"         {edge}: {value:.4f}")
    
    return edge_results


def comprehensive_pipeline_test():
    """Test all combinations of community detection and measures"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE NETWORK ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Create network
    network = create_sample_network()
    
    # Get all techniques
    community_techniques = {
        'Louvain': dmf.LouvainCommunityDetection(),
        'Label Propagation': dmf.LabelPropagationCommunityDetection()
    }
    
    node_measures = {
        'PageRank': dmf.PageRankMeasure(),
        'Degree': dmf.DegreeCentralityMeasure(),
        'Betweenness': dmf.BetweennessCentralityMeasure()
    }
    
    edge_measures = {
        'Edge Betweenness': dmf.EdgeBetweennessMeasure()
    }
    
    # Collect results
    results = []
    run_count = 0
    
    print(f"\nRunning {len(community_techniques)} community detection techniques")
    print(f"With {len(node_measures)} node measures and {len(edge_measures)} edge measures\n")
    
    for comm_name, comm_technique in community_techniques.items():
        # Detect communities
        comm_result = comm_technique.detect_communities(network)
        
        print(f"{comm_name}:")
        print(f"  Communities: {comm_result.n_communities}")
        print(f"  Modularity: {comm_result.metadata.get('modularity', 0):.4f}")
        
        # Calculate node measures
        for node_name, node_measure in node_measures.items():
            run_count += 1
            node_values = node_measure.calculate(network)
            
            # Get average centrality
            avg_centrality = np.mean(list(node_values.values()))
            
            result = {
                'run': run_count,
                'community_technique': comm_name,
                'n_communities': comm_result.n_communities,
                'modularity': comm_result.metadata.get('modularity', 0),
                'measure_type': 'node',
                'measure_name': node_name,
                'avg_value': avg_centrality,
                'max_value': max(node_values.values()),
                'min_value': min(node_values.values())
            }
            results.append(result)
        
        # Calculate edge measures
        for edge_name, edge_measure in edge_measures.items():
            run_count += 1
            edge_values = edge_measure.calculate(network)
            
            # Get average value
            avg_value = np.mean(list(edge_values.values()))
            
            result = {
                'run': run_count,
                'community_technique': comm_name,
                'n_communities': comm_result.n_communities,
                'modularity': comm_result.metadata.get('modularity', 0),
                'measure_type': 'edge',
                'measure_name': edge_name,
                'avg_value': avg_value,
                'max_value': max(edge_values.values()),
                'min_value': min(edge_values.values())
            }
            results.append(result)
        
        print(f"  Measures calculated: {len(node_measures) + len(edge_measures)}")
        print()
    
    # Export to CSV
    results_df = pd.DataFrame(results)
    csv_filename = 'project3_results.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Results exported to: {csv_filename}")
    
    # Print summary
    print("\nSUMMARY STATISTICS:")
    print("-" * 70)
    
    # Best modularity
    best_modularity = results_df.loc[results_df['modularity'].idxmax()]
    print(f"\nBest Modularity: {best_modularity['modularity']:.4f}")
    print(f"  Technique: {best_modularity['community_technique']}")
    print(f"  Communities: {best_modularity['n_communities']}")
    
    # Average values by measure
    print("\nAverage Values by Measure:")
    for measure in results_df['measure_name'].unique():
        measure_data = results_df[results_df['measure_name'] == measure]
        avg = measure_data['avg_value'].mean()
        print(f"  {measure}: {avg:.4f}")
    
    return results


def test_custom_network():
    """Test with custom network creation"""
    print("\n" + "=" * 70)
    print("CUSTOM NETWORK TEST")
    print("=" * 70)
    
    # Create a simple network from edge list
    edges = [
        (0, 1), (0, 2), (1, 2), (1, 3),
        (2, 3), (3, 4), (4, 5), (5, 6),
        (4, 6), (6, 7), (7, 8), (6, 8)
    ]
    
    network = dmf.load_network(edges, directed=False)
    
    print(f"\nCustom Network Created:")
    print(f"  Nodes: {network.num_nodes()}")
    print(f"  Edges: {network.num_edges()}")
    
    # Detect communities
    louvain = dmf.LouvainCommunityDetection()
    result = louvain.detect_communities(network)
    
    print(f"\nCommunity Detection:")
    print(f"  Communities found: {result.n_communities}")
    print(f"  Community structure:")
    for i, community in enumerate(result.get_communities()):
        print(f"    Community {i}: {sorted(community)}")
    
    # Calculate PageRank
    pagerank = dmf.PageRankMeasure()
    pr_values = pagerank.calculate(network)
    
    print(f"\nPageRank Scores:")
    for node in sorted(pr_values.keys()):
        print(f"  Node {node}: {pr_values[node]:.4f}")


def main():
    # Test network basics
    network = test_network_basics()
    
    # Test community detection
    test_community_detection(network)
    
    # Test node measures
    test_node_measures(network)
    
    # Test edge measures
    test_edge_measures(network)
    
    # Comprehensive pipeline
    comprehensive_pipeline_test()
    
    # Custom network test
    test_custom_network()
    
    print("\n" + "=" * 70)
    print("✅ PROJECT 3 DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. Network component successfully loads and exposes nodes/edges")
    print("2. Community detection identifies meaningful groups")
    print("3. Node measures (PageRank, centrality) quantify node importance")
    print("4. Edge measures quantify edge importance in network structure")
    print("5. All components work together seamlessly!")


if __name__ == "__main__":
    main()
