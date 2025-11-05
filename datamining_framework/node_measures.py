import networkx as nx
from .core import NodeMeasure


class PageRankMeasure(NodeMeasure):
    
    # PageRank centrality measure.

    
    def calculate(self, network, **params):

        alpha = params.get('alpha', 0.85)
        max_iter = params.get('max_iter', 100)
        tol = params.get('tol', 1e-6)
        
        graph = network.get_graph()
        
        pagerank = nx.pagerank(
            graph,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol
        )
        
        return pagerank


class DegreeCentralityMeasure(NodeMeasure):
    
    # Degree centrality measure.

    
    def calculate(self, network, **params):

        graph = network.get_graph()
        
        degree_centrality = nx.degree_centrality(graph)
        
        return degree_centrality


class BetweennessCentralityMeasure(NodeMeasure):
    
    # Betweenness centrality measure for nodes.

    
    def calculate(self, network, **params):

        normalized = params.get('normalized', True)
        k = params.get('k', None)
        
        graph = network.get_graph()
        
        betweenness = nx.betweenness_centrality(
            graph,
            normalized=normalized,
            k=k
        )
        
        return betweenness


class ClosenessCentralityMeasure(NodeMeasure):
    
    # Closeness centrality measure.

    
    def calculate(self, network, **params):

        wf_improved = params.get('wf_improved', True)
        
        graph = network.get_graph()
        
        closeness = nx.closeness_centrality(
            graph,
            wf_improved=wf_improved
        )
        
        return closeness
