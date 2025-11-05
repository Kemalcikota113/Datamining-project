import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from .core import DistanceMeasure


class EuclideanDistance(DistanceMeasure):
    
    # Euclidean distance measure using sklearn implementation.

    
    def calculate(self, point1, point2):
        point1 = np.array(point1).reshape(1, -1)
        point2 = np.array(point2).reshape(1, -1)
        return euclidean_distances(point1, point2)[0, 0]


class ManhattanDistance(DistanceMeasure):

    # Manhattan distance measure using sklearn implementation.

    
    def calculate(self, point1, point2):
        point1 = np.array(point1).reshape(1, -1)
        point2 = np.array(point2).reshape(1, -1)
        return manhattan_distances(point1, point2)[0, 0]


class CosineDistance(DistanceMeasure):
    
    # Cosine distance measure using sklearn implementation.

    
    def calculate(self, point1, point2):
        point1 = np.array(point1).reshape(1, -1)
        point2 = np.array(point2).reshape(1, -1)
        return cosine_distances(point1, point2)[0, 0]
