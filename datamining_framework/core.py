import pandas as pd
import numpy as np


class Dataset:
    # Handle structured data with methods to access points and features
    
    def __init__(self, data):
        if isinstance(data, str):
            if data.endswith('.csv'):
                self.data = pd.read_csv(data)
            elif data.endswith('.xlsx') or data.endswith('.xls'):
                self.data = pd.read_excel(data)
            else:
                raise ValueError("Unsupported file format")
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Unsupported data type")
    
    def get_data_points(self):
        # Return data as numpy array.
        return self.data.values
    
    def get_features(self):
        # Return list of feature names.
        return self.data.columns.tolist()
    
    def get_shape(self):
        #Return dataset shape.
        return self.data.shape
    
    def get_dataframe(self):
        #Return copy of underlying DataFrame
        return self.data.copy()


class DistanceMeasure:
    #Base class for distance measures.
    
    def calculate(self, point1, point2):
        #Calculate distance between two points.
        raise NotImplementedError


class ClusteringTechnique:
    #Base class for clustering techniques
    
    def cluster(self, dataset, distance_measure=None, **hyperparams):
        #Perform clustering on dataset.
        raise NotImplementedError


class QualityMeasure:
    #Base class for quality measures.
    
    def evaluate(self, clustering_result, dataset):
        #Evaluate clustering quality.
        raise NotImplementedError


class ClusteringResult:
    #Container for clustering results.
    
    def __init__(self, labels, centers=None, metadata=None):
        self.labels = np.array(labels)
        self.centers = centers
        self.metadata = metadata or {}
        self.n_clusters = len(np.unique(labels[labels >= 0]))
    
    def get_labels(self):
        #Return cluster labels.
        return self.labels
    
    def get_centers(self):
        #Return cluster centers.
        return self.centers
    
    def get_clusters(self):
        #Return list of clusters (each cluster is list of point indices).
        clusters = []
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            if label >= 0:
                cluster_points = np.where(self.labels == label)[0].tolist()
                clusters.append(cluster_points)
        
        return clusters


class DimensionalityReductionTechnique:
    #Base class for dimensionality reduction techniques.
    
    def reduce(self, dataset, distance_measure=None, **hyperparams):
        #Perform dimensionality reduction on dataset.
        raise NotImplementedError


class DRQualityMeasure:
    #Base class for DR quality measures.
    
    def evaluate(self, dr_result, original_dataset):
        #Evaluate dimensionality reduction quality.
        raise NotImplementedError


class DRResult:
    #Container for dimensionality reduction results.
    
    def __init__(self, reduced_data, explained_variance=None, metadata=None):
        self.reduced_data = np.array(reduced_data)
        self.explained_variance = explained_variance
        self.metadata = metadata or {}
        self.n_components = self.reduced_data.shape[1] if len(self.reduced_data.shape) > 1 else 1
    
    def get_reduced_data(self):
        #Return reduced data as numpy array.
        return self.reduced_data
    
    def get_reduced_dataset(self):
        #Return reduced data as Dataset object
        return Dataset(self.reduced_data)
    
    def get_explained_variance(self):
        #Return explained variance if available.
        return self.explained_variance


class Network:
    #Handle network/graph data with methods to access nodes and edges.
    
    def __init__(self, data=None, directed=False):
        import networkx as nx
        
        self.directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()
        
        if data is not None:
            if isinstance(data, str):
                self._load_from_file(data)
            elif isinstance(data, list):
                self.graph.add_edges_from(data)
            elif isinstance(data, np.ndarray):
                self._load_from_adjacency_matrix(data)
            else:
                raise ValueError("Unsupported data type for Network")
    
    def _load_from_file(self, filepath):
        #Load network from file.
        import networkx as nx
        
        if filepath.endswith('.edgelist') or filepath.endswith('.txt'):
            self.graph = nx.read_edgelist(filepath, create_using=nx.DiGraph() if self.directed else nx.Graph())
        elif filepath.endswith('.gml'):
            self.graph = nx.read_gml(filepath)
        elif filepath.endswith('.graphml'):
            self.graph = nx.read_graphml(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            if len(df.columns) >= 2:
                edges = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
                self.graph.add_edges_from(edges)
            else:
                raise ValueError("CSV must have at least 2 columns for edge list")
        else:
            raise ValueError("Unsupported file format for Network")
    
    def _load_from_adjacency_matrix(self, matrix):
        #Load network from adjacency matrix.
        import networkx as nx
        self.graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph() if self.directed else nx.Graph())
    
    def get_nodes(self):
        #Return list of nodes.
        return list(self.graph.nodes())
    
    def get_edges(self):
        #Return list of edges.
        return list(self.graph.edges())
    
    def get_graph(self):
        #Return underlying NetworkX graph.
        return self.graph
    
    def get_adjacency_matrix(self):
        #Return adjacency matrix as numpy array.
        import networkx as nx
        return nx.to_numpy_array(self.graph)
    
    def num_nodes(self):
        #Return number of nodes.
        return self.graph.number_of_nodes()
    
    def num_edges(self):
        #Return number of edges.
        return self.graph.number_of_edges()
    
    def is_directed(self):
        #Return True if network is directed.
        return self.directed


class CommunityDetectionTechnique:
    #Base class for community detection techniques.

    def detect_communities(self, network, **hyperparams):
        #Detect communities in a network.
        raise NotImplementedError


class NodeMeasure:
    #Base class for node measures.

    def calculate(self, network, **params):
        #Calculate measure for each node in the network.
        raise NotImplementedError


class EdgeMeasure:
    #Base class for edge measures.

    def calculate(self, network, **params):
        #Calculate measure for each edge in the network.
        raise NotImplementedError


class CommunityResult:
    #Container for community detection results.

    def __init__(self, communities, metadata=None):
        self.metadata = metadata or {}
        
        # Support both dict and list formats
        if isinstance(communities, dict):
            self.node_to_community = communities
            community_sets = {}
            for node, comm_id in communities.items():
                if comm_id not in community_sets:
                    community_sets[comm_id] = []
                community_sets[comm_id].append(node)
            self.communities = list(community_sets.values())
        else:
            self.communities = [list(c) for c in communities]
            self.node_to_community = {}
            for comm_id, community in enumerate(self.communities):
                for node in community:
                    self.node_to_community[node] = comm_id
        
        self.n_communities = len(self.communities)
    
    def get_communities(self):
        #Return list of communities.
        return self.communities
    
    def get_community_labels(self):
        #Return node to community label mapping.
        return self.node_to_community
    
    def get_community_of_node(self, node):
        #Return community label for a specific node.
        return self.node_to_community.get(node, -1)
