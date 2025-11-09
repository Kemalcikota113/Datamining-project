import networkx as nx
from networkx.algorithms import community
from .core import CommunityDetectionTechnique, CommunityResult


class LouvainCommunityDetection(CommunityDetectionTechnique):

    # Louvain method for community detection.

    
    def detect_communities(self, network, **hyperparams):

        resolution = hyperparams.get('resolution', 1.0)
        random_state = hyperparams.get('random_state', 42)
        
        graph = network.get_graph()
        
        # Apply Louvain algorithm
        communities_gen = community.louvain_communities(
            graph, 
            resolution=resolution,
            seed=random_state
        )
        
        communities = [list(c) for c in communities_gen]
        
        # Calculate modularity
        modularity = community.modularity(graph, communities_gen)
        
        metadata = {
            'algorithm': 'Louvain',
            'hyperparams': hyperparams,
            'modularity': modularity
        }
        
        return CommunityResult(communities, metadata=metadata)


class GirvanNewmanCommunityDetection(CommunityDetectionTechnique):
    
    # Girvan-Newman method for community detection.

    
    def detect_communities(self, network, **hyperparams):

        k = hyperparams.get('k', None)
        
        graph = network.get_graph()
        
        # Apply Girvan-Newman algorithm
        communities_generator = community.girvan_newman(graph)
        
        # Get communities at different levels
        if k is not None:
            # Get specific number of communities
            for _ in range(k - 1):
                communities_tuple = next(communities_generator)
            communities = [list(c) for c in communities_tuple]
        else:
            # Find best partition based on modularity
            best_communities = None
            best_modularity = -1
            
            limited_iter = 0
            max_iterations = min(10, graph.number_of_nodes() - 1)
            
            for communities_tuple in communities_generator:
                limited_iter += 1
                if limited_iter > max_iterations:
                    break
                    
                mod = community.modularity(graph, communities_tuple)
                if mod > best_modularity:
                    best_modularity = mod
                    best_communities = communities_tuple
            
            communities = [list(c) for c in best_communities] if best_communities else [[n] for n in graph.nodes()]
        
        # Calculate final modularity
        final_modularity = community.modularity(graph, communities)
        
        metadata = {
            'algorithm': 'Girvan-Newman',
            'hyperparams': hyperparams,
            'modularity': final_modularity,
            'n_communities': len(communities)
        }
        
        return CommunityResult(communities, metadata=metadata)


class FastNewmanCommunityDetection(CommunityDetectionTechnique):
    
    # Fast greedy modularity maximization (Clauset-Newman-Moore algorithm).

    
    def detect_communities(self, network, **hyperparams):

        graph = network.get_graph()
        
        # Apply fast greedy modularity maximization
        communities_gen = community.greedy_modularity_communities(graph)
        communities = [list(c) for c in communities_gen]
        
        # Calculate modularity
        modularity = community.modularity(graph, communities_gen)
        
        metadata = {
            'algorithm': 'Fast Newman',
            'hyperparams': hyperparams,
            'modularity': modularity,
            'n_communities': len(communities)
        }
        
        return CommunityResult(communities, metadata=metadata)
