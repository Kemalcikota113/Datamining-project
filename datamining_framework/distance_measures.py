"""
Distance Measures Implementation
Three different distance measure implementations as required.
"""

import numpy as np
from .core import DistanceMeasure


class EuclideanDistance(DistanceMeasure):
    """
    Euclidean distance measure.
    Standard L2 norm distance.
    """
    
    def calculate(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First data point (array-like)
            point2: Second data point (array-like)
            
        Returns:
            float: Euclidean distance
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sqrt(np.sum((point1 - point2) ** 2))


class ManhattanDistance(DistanceMeasure):
    """
    Manhattan distance measure.
    L1 norm distance (sum of absolute differences).
    """
    
    def calculate(self, point1, point2):
        """
        Calculate Manhattan distance between two points.
        
        Args:
            point1: First data point (array-like)
            point2: Second data point (array-like)
            
        Returns:
            float: Manhattan distance
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sum(np.abs(point1 - point2))


class CosineDistance(DistanceMeasure):
    """
    Cosine distance measure.
    1 - cosine similarity between two vectors.
    """
    
    def calculate(self, point1, point2):
        """
        Calculate Cosine distance between two points.
        
        Args:
            point1: First data point (array-like)
            point2: Second data point (array-like)
            
        Returns:
            float: Cosine distance (0 to 2)
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        # Handle zero vectors
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_similarity = np.dot(point1, point2) / (norm1 * norm2)
        # Clamp to [-1, 1] to handle numerical errors
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        return 1.0 - cosine_similarity
