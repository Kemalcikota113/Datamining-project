import networkx as nx
from .core import EdgeMeasure


class EdgeBetweennessMeasure(EdgeMeasure):
    
    # Edge betweenness centrality measure.

    
    def calculate(self, network, **params):

        normalized = params.get('normalized', True)
        k = params.get('k', None)
        
        graph = network.get_graph()
        
        edge_betweenness = nx.edge_betweenness_centrality(
            graph,
            normalized=normalized,
            k=k
        )
        
        return edge_betweenness


class EdgeCurrentFlowMeasure(EdgeMeasure):
    
    # Edge current flow betweenness centrality.

    
    def calculate(self, network, **params):

        normalized = params.get('normalized', True)
        
        graph = network.get_graph()
        
        # Current flow requires undirected graph
        if network.is_directed():
            graph = graph.to_undirected()
        
        # Check if graph is connected
        if not nx.is_connected(graph):
            # For disconnected graphs, compute for largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()
        
        try:
            edge_current_flow = nx.edge_current_flow_betweenness_centrality(
                graph,
                normalized=normalized
            )
        except:
            # Fallback to regular edge betweenness if current flow fails
            edge_current_flow = nx.edge_betweenness_centrality(
                graph,
                normalized=normalized
            )
        
        return edge_current_flow


class EdgeLoadMeasure(EdgeMeasure):
    
    # Edge load centrality measure.
    
    
    def calculate(self, network, **params):

        cutoff = params.get('cutoff', None)
        
        graph = network.get_graph()
        
        try:
            edge_load = nx.edge_load_centrality(graph, cutoff=cutoff)
        except:
            # Fallback to edge betweenness if load centrality fails
            edge_load = nx.edge_betweenness_centrality(graph)
        
        return edge_load


class EdgeWeightMeasure(EdgeMeasure):
    
    # Edge weight measure.

    
    def calculate(self, network, **params):

        weight_attr = params.get('weight_attr', 'weight')
        
        graph = network.get_graph()
        edge_weights = {}
        
        for edge in graph.edges():
            # Get edge weight, default to 1.0 if not present
            weight = graph[edge[0]][edge[1]].get(weight_attr, 1.0)
            edge_weights[edge] = weight
        
        return edge_weights
