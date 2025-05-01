import numpy as np
import math
from typing import Any, Tuple
from MultisetEmbedder import MultisetEmbedder

class DottingEncoder:
    """
    Encodes via sorted length-n lists of dot products.
    The first k-such lists are with standard coordinate axes.
    After that come "extra_dots" extra dot products.
    """

    def __init__(self, k: int, extra_dots: int):
        self.k = k
        self.extra_dots = extra_dots
        self.total_dots = k + extra_dots
        
        # Create random unit vectors for the extra dot products
        if extra_dots > 0:
            # Generate random matrix with shape (extra_dots, k)
            random_matrix = np.random.randn(extra_dots, k)
            
            # Normalize each row to create unit vectors
            norms = np.linalg.norm(random_matrix, axis=1, keepdims=True)
            unit_vectors = random_matrix / norms
            
            # Combine standard basis with random unit vectors
            if k > 0:
                standard_basis = np.eye(k)
                self.matrix = np.vstack([standard_basis, unit_vectors])
            else:
                self.matrix = unit_vectors
        else:
            # Just use the standard basis
            self.matrix = np.eye(k)

    def encode(self, data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Any]:
        """
        Encode batched data of shape (b, n, k) using dot products.
        
        Args:
            data: Input array of shape (b, n, k)
            
        Returns:
            Tuple of (embeddings, (n, k), metadata)
            embeddings: Array of shape (b, n * total_dots)
            (n, k): Input dimensions
            metadata: Additional information (None in this implementation)
        """
        b, n, k = data.shape
        
        # Verify dimensions match
        if k != self.k:
            raise ValueError(f"Data has {k} features but encoder was created with {self.k} features")
        
        # Initialize result array
        result = np.zeros((b, n * self.total_dots), dtype=np.float64)
        
        # Process each batch element
        for batch_idx in range(b):
            batch_data = data[batch_idx]
            
            # Compute dot products with each vector in the matrix
            # Shape: (n, total_dots)
            dot_products = np.dot(batch_data, self.matrix.T)
            
            # Sort each column of dot products
            # Shape: (n, total_dots) with each column sorted
            sorted_dot_products = np.sort(dot_products, axis=0)
            
            # Flatten the result
            # Shape: (n * total_dots,)
            flattened_dots = sorted_dot_products.flatten()
            
            # Store in result array
            result[batch_idx] = flattened_dots
        
        metadata = None
        return result, (n, k), metadata

    def size_from_n_k(self, n: int, k: int) -> int:
        """Calculate embedding dimension from n and k"""
        return n * self.total_dots


class Embedder(MultisetEmbedder):
    """
    This encoder encodes via sorted length-n lists of dot products.
    The first k-such lists are with standard coordinate axes.
    After that come "extra_dots" extra dot products. extra_dots is set in the constructor.
    """

    def __init__(self, n: int = None, k: int = None):
        super().__init__()
        self.n = n
        self.k = k
        
        if n is not None and k is not None:
            if k < 0 or n < 0:
                raise ValueError("It makes no sense to claim we will code vectors or sets for k<0.")

            if k == 0 or n == 0:
                raise ValueError("We choose not to undertake to embed non-data!")

            M_min = (k - 1) * (math.floor(math.log2(n)) + 1) + 1  # Conjectured to be sufficient
            extra_dots = M_min - k 
            assert extra_dots >= 0
            assert k == 1 or extra_dots > 0

            self._dotting_encoder = DottingEncoder(k=k, extra_dots=extra_dots)

    def embed_generic(self, data: np.ndarray, debug=False) -> Tuple[np.ndarray, Any]:
        """
        Main embedding method for the general case.
        
        Args:
            data: Input array of shape (b, n, k)
            debug: Whether to print debug information
            
        Returns:
            Tuple of (embedding, metadata)
            embedding: Array of shape (b, embedding_dim)
            metadata: Additional information (None in this implementation)
        """
        b, n, k = data.shape
        
        # Initialize dotting encoder if not already initialized
        if self.n is None or self.k is None:
            self.n = n
            self.k = k
            
            M_min = (k - 1) * (math.floor(math.log2(n)) + 1) + 1
            extra_dots = M_min - k 
            assert extra_dots >= 0
            assert k == 1 or extra_dots > 0
            
            self._dotting_encoder = DottingEncoder(k=k, extra_dots=extra_dots)
        
        # Verify dimensions match
        expected_size = self.size_from_n_k(n, k)
        if expected_size == -1:
            raise ValueError(f"We do not undertake to embed data of shape {data.shape}")
        
        if debug:
            print(f"data is {data}")
        
        embedding, size, metadata = self._dotting_encoder.encode(data)
        
        if debug:
            print(f"Embedding is {embedding} with shape {embedding.shape}")
        
        assert embedding.shape[1] == expected_size
        return embedding, metadata
    
    def size_from_n_k_generic(self, n: int, k: int) -> int:
        """Calculate embedding dimension from n and k"""
        if n != self.n or k != self.k:
            return -1  # We are optimized to work with a certain n and a certain k
        return self._dotting_encoder.size_from_n_k(n, k)

    def embed_kOne(self, data: np.ndarray, debug=False) -> Tuple[np.ndarray, Any]:
        """Handle the special case where k=1 (each vector is 1D)"""
        metadata = None
        return MultisetEmbedder.embed_kOne_sorting(data), metadata