import torch
from typing import Any, Tuple, Optional, Union

class TorchMultisetEmbedder(torch.nn.Module):
    """
    This is a base class for objects which embed length-n sets of k-vectors using PyTorch.

    Encoders are not necessarily embedders since encoders do not need to be injective. 
    All embedders are encoders, however.

    Strictly speaking these are "multisets" not "sets" since the sets can hold repeated 
    objects and retain knowledge of the number of repeats.  However, we are sometimes guilty 
    of abbreviating "multiset" to just "set".

    The set to be embed should be inputs as a 3D torch tensor with shape (b,n,k).
    The order of the vectors within the tensor can be arbitrary.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = torch.device('cpu'), dtype: Optional[torch.dtype] = torch.float32):
        """
        Initialize the embedder.
        
        Args:
            device: The device to use for computations (e.g., 'cpu', 'cuda:0')
            dtype: The data type to use for computations (e.g., torch.float32, torch.float64)
        """
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32

    def forward(self, data: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, Tuple[int, int], Any]:
        """
        Forward pass of the embedder.
        
        Args:
            data: Input tensor of shape (b,n,k)
            debug: Whether to print debug information
            
        Returns:
            Tuple of (embedding, (n,k), metadata)
            embedding: Tensor of shape (b, embedding_dim)
            (n,k): Tuple of input dimensions
            metadata: Additional information (None in base implementation)
        """
        # Ensure input is on correct device and dtype
        data = data.to(device=self.device, dtype=self.dtype)
        
        b, n, k = data.shape
        expected_order = self.size_from_n_k(n, k)

        if n < 0 or k < 0:
            raise ValueError("Tensors should not have negative sizes!")
        if n == 0 or k == 0:
            return torch.empty((b, 0), device=self.device, dtype=self.dtype), (n, k), None
        if n == 1:
            embedding = data.reshape(b, -1)  # This implementation is a coverall
            assert embedding.shape == (b, k)
            assert embedding.shape[-1] == expected_order
            return embedding, (n, k), None
        if k == 1:
            assert k == 1 and n >= 0  # Preconditions for calling self.embed_kOne
            assert self.is_kOne_n_k(n, k)  # Precondition for calling self.embed_kOne
            embedding, metadata = self.embed_kOne(data, debug)
            assert embedding.shape == (b, n)
            assert embedding.shape[-1] == expected_order
            return embedding, (n, k), metadata

        assert n > 1 and k > 1  # Preconditions for calling self.embed_generic
        assert self.is_generic_n_k(n, k)  # Precondition for calling self.embed_generic
        embedding, metadata = self.embed_generic(data, debug)
        assert embedding.shape[-1] == expected_order
        return embedding, (n, k), metadata

    def size_from_n_k(self, n: int, k: int) -> int:
        """
        Returns the number of reals that the embedding would contain if a set containing 
        n "k-vectors" were to be embedded. Returns -1 if embedding is not possible.
        """
        if n < 0 or k < 0:
            return -1
        if n == 0 or k == 0:
            return 0
        if n == 1:
            return k
        if k == 1:
            return n
        return self.size_from_n_k_generic(n, k)

    def size_from_tensor(self, data: torch.Tensor) -> int:
        """
        Returns the number of reals that the embedding would contain if the set represented 
        by "data" were to be embedded. Returns -1 if data is not encodable.
        """
        _, n, k = data.shape
        return self.size_from_n_k(n, k)

    def embed_kOne(self, data: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, Any]:
        """
        Derived classes should implement this method.
        This method should OPTIMALLY embed data for which n>=0 and k==1.
        """
        raise NotImplementedError()

    def embed_generic(self, data: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, Any]:
        """
        Derived classes should implement this method.
        This method should embed data for which n>1 and k>1.
        """
        raise NotImplementedError()

    def size_from_n_k_generic(self, n: int, k: int) -> int:
        """
        Derived classes should implement this method.
        This method should report the embedding size for data for which n>1 and k>1.
        """
        raise NotImplementedError()

    def test_me(self):
        _ = self.size_from_n_k_generic(2, 2)  # Check implementation exists
        test_tensor = torch.tensor([[[1., 2.], [3., 4.]]], device=self.device, dtype=self.dtype)
        _ = self.embed_generic(test_tensor)  # Check implementation exists
        test_tensor = torch.tensor([[[1.], [3.]]], device=self.device, dtype=self.dtype)
        _ = self.embed_kOne(test_tensor)  # Check implementation exists

    @staticmethod
    def embed_kOne_sorting(data: torch.Tensor) -> torch.Tensor:
        assert TorchMultisetEmbedder.is_kOne_data(data)
        return torch.sort(data.reshape((data.shape[0], -1)), dim=1)[0]

    @staticmethod
    def is_kOne_data(data: torch.Tensor) -> bool:
        _, n, k = data.shape
        return TorchMultisetEmbedder.is_kOne_n_k(n, k)

    @staticmethod
    def is_kOne_n_k(n: int, k: int) -> bool:
        return n >= 0 and k == 1

    @staticmethod
    def is_generic_data(data: torch.Tensor) -> bool:
        _, n, k = data.shape
        return TorchMultisetEmbedder.is_generic_n_k(n, k)

    @staticmethod
    def is_generic_n_k(n: int, k: int) -> bool:
        return n > 1 and k > 1 