import torch
import math
from typing import Any, Tuple, Optional, Union
from torch_multiset_embedder import TorchMultisetEmbedder

class TorchDottingEmbedder(TorchMultisetEmbedder):
    """
    A combined embedder that encodes via sorted length-n lists of dot products.
    The first k-such lists are with standard coordinate axes.
    After that come "extra_dots" extra dot products.
    """

    def __init__(self, n: Optional[int] = None, k: Optional[int] = None, 
                 device: Optional[Union[str, torch.device]] = torch.device('cpu'), 
                 dtype: Optional[torch.dtype] = torch.float32):
        """
        Initialize the embedder.
        
        Args:
            n: Number of vectors in each set
            k: Dimension of each vector
            device: The device to use for computations (e.g., 'cpu', 'cuda:0')
            dtype: The data type to use for computations (e.g., torch.float32, torch.float64)
        """
        super().__init__(device=device, dtype=dtype)
        self.n = None
        self.k = None
        self._matrix = None
        self._matrix_shape = None  # Cache for matrix shape
        if n is not None and k is not None:
            self._ensure_initialized(n, k)

    def _ensure_initialized(self, n: int, k: int):
        if self.n is not None and self.k is not None:
            return  # Already initialized
        self.n = n
        self.k = k
        if k < 0 or n < 0:
            raise ValueError("It makes no sense to claim we will code vectors or sets for k<0.")
        if k == 0 or n == 0:
            raise ValueError("We choose not to undertake to embed non-data!")
        M_min = (k - 1) * (math.floor(math.log2(n)) + 1) + 1  # Conjectured to be sufficient
        extra_dots = M_min - k 
        assert extra_dots >= 0
        assert n == 1 or k == 1 or extra_dots > 0
        self._initialize_matrix(k, extra_dots)

    def _initialize_matrix(self, k: int, extra_dots: int):
        """Initialize the matrix of reference vectors for dot products"""
        if extra_dots > 0:
            # Pre-allocate the full matrix
            total_dots = k + extra_dots
            
            # Create standard basis first
            if k > 0:
                standard_basis = torch.eye(k, device=self.device, dtype=self.dtype)
            
            # Generate random vectors and normalize them
            random_vectors = torch.randn(extra_dots, k, device=self.device, dtype=self.dtype)
            norms = torch.sqrt(torch.sum(random_vectors * random_vectors, dim=1, keepdim=True))
            random_vectors = random_vectors / norms
            
            # Combine standard basis and random vectors
            self._matrix = torch.nn.Parameter(torch.cat([standard_basis, random_vectors], dim=0))
        else:
            # Just use the standard basis
            self._matrix = torch.nn.Parameter(torch.eye(k, device=self.device, dtype=self.dtype))
        
        self._matrix_shape = self._matrix.shape

    def embed_generic(self, data: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, Any]:
        """
        Main embedding method for the general case.
        
        Args:
            data: Input tensor of shape (b, n, k)
            debug: Whether to print debug information
            
        Returns:
            Tuple of (embedding, metadata)
            embedding: Tensor of shape (b, embedding_dim)
            metadata: Additional information (None in this implementation)
        """
        b, n, k = data.shape
        # Ensure initialization
        self._ensure_initialized(n, k)
        # Verify dimensions match
        expected_size = self.size_from_n_k_generic(n, k)
        if expected_size == -1:
            raise ValueError(f"We do not undertake to embed data of shape {data.shape}")
        # Compute dot products for entire batch at once
        # Shape: (b, n, total_dots)
        dot_products = torch.matmul(data, self._matrix.T)
        # Sort each column of dot products for each batch element
        # Shape: (b, n, total_dots) with each column sorted
        sorted_dot_products = torch.sort(dot_products, dim=1)[0]
        # Reshape to final output shape (b, n * total_dots)
        result = sorted_dot_products.reshape(b, -1)
        metadata = None
        return result, metadata

    def embed_kOne(self, data: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, Any]:
        """Handle the special case where k=1 (each vector is 1D)"""
        metadata = None
        return TorchMultisetEmbedder.embed_kOne_sorting(data), metadata

    def size_from_n_k_generic(self, n: int, k: int) -> int:
        """Calculate embedding dimension from n and k"""
        if n != self.n or k != self.k:
            return -1  # We are optimized to work with a certain n and a certain k
        return n * self._matrix_shape[0] if self._matrix_shape is not None else -1 