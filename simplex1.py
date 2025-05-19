import numpy as np
from itertools import pairwise
from dataclasses import dataclass, field
from typing import Self, Any, Tuple
import hashlib
from tools import sort_np_array_rows_lexicographically
from np_MultisetEmbedder import MultisetEmbedder

class Embedder(MultisetEmbedder):
    """
    Embedder class for batched permutation-invariant encoding of vector sets.
    Processes inputs of shape (b,n,k) where:
    - b is the batch size
    - n is the number of vectors in each set
    - k is the dimension of each vector
    """

    def embed_kOne(self, data: np.ndarray, debug=False) -> tuple[np.ndarray, Any]:
        """Handle the special case where k=1 (each vector is 1D)"""
        metadata = None
        return MultisetEmbedder.embed_kOne_sorting(data), metadata

    def embed_generic(self, data: np.ndarray, debug=False) -> tuple[np.ndarray, Any]:
        """
        Main embedding method for the general case.
        
        Args:
            data: Input array of shape (b,n,k)
            debug: Whether to print debug information
            
        Returns:
            Tuple of (embedding, metadata)
            embedding: Array of shape (b, embedding_dim)
            metadata: Additional information (None in this implementation)
        """
        assert MultisetEmbedder.is_generic_data(data)  # Precondition
        b, n, k = data.shape
        
        # Initialize output array
        embed_dim = self.size_from_n_k(n, k)
        result = np.zeros((b, embed_dim), dtype=np.float64)
        
        # Process each batch item separately
        for batch_idx in range(b):
            batch_data = data[batch_idx]
            
            # Flatten data into (value, (j,i)) pairs for sorting
            flattened_data = [(batch_data[j][i], (j, i)) for j in range(n) for i in range(k)]
            sorted_data = sorted(flattened_data, key=lambda x: -x[0])
            
            # Extract min and max elements
            min_element = sorted_data[-1][0] if sorted_data else 0
            max_element = sorted_data[0][0] if sorted_data else 0
            
            # Calculate pairwise differences between adjacent sorted elements
            difference_data = [(x[0] - y[0], x[1]) for x, y in pairwise(sorted_data)]
            
            # Construct Maximal Simplex Vertices
            difference_data_with_MSVs = [
                (delta, Maximal_Simplex_Vertex({eji for _, eji in difference_data[0:i + 1]})) 
                for i, (delta, _) in enumerate(difference_data)
            ]
            
            # Sort by delta values in descending order
            sorted_difference_data_with_MSVs = sorted(difference_data_with_MSVs, key=lambda x: -x[0])
            
            # Extract deltas and MSVs in current order
            deltas_in_current_order = [delta for delta, _ in sorted_difference_data_with_MSVs]
            msvs_in_current_order = [msv for _, msv in sorted_difference_data_with_MSVs]
            
            expected_number_of_vertices = n * k - 1
            assert len(deltas_in_current_order) == expected_number_of_vertices
            assert len(msvs_in_current_order) == expected_number_of_vertices
            
            # Calculate barycentric subdivision coordinates
            difference_data_in_subdivided_simplex = [
                ((i + 1) * (deltas_in_current_order[i] - 
                    (deltas_in_current_order[i + 1] if i + 1 < expected_number_of_vertices else 0)),
                 Eji_LinComb(n, k, msvs_in_current_order[:i + 1])) 
                for i in range(expected_number_of_vertices)
            ]
            
            # Get canonical form of difference data
            canonical_difference_data = [(delta, msv.get_canonical_form()) for (delta, msv) in difference_data_in_subdivided_simplex]
            
            # Calculate embedding dimension
            bigN = 2 * (n * k - 1) + 1
            
            # Map to points in unit hypercube and compute embedding
            difference_point_pairs = [(delta, eji_lin_com.hash_to_point_in_unit_hypercube(bigN)) 
                                     for (delta, eji_lin_com) in canonical_difference_data]
            
            first_half_of_embedding = sum([delta * point for delta, point in difference_point_pairs]) + np.zeros(bigN)
            
            # Assemble the full embedding
            result[batch_idx, :bigN] = first_half_of_embedding
            result[batch_idx, -1] = min_element
            result[batch_idx, -2] = max_element
            
        metadata = None
        return result, metadata
    
    def size_from_n_k_generic(self, n: int, k: int) -> int:
        """Calculate embedding dimension from n and k"""
        return 2 * n * k + 1


@dataclass
class Maximal_Simplex_Vertex:
    """Represents a maximal simplex vertex in the complex"""
    _vertex_set: set[tuple] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self._vertex_set)

    def __iter__(self):
        return iter(self._vertex_set)

    def get_canonical_form(self):
        """
        Get canonical form by sorting by i index and renumbering j indices.
        This mods out by Sn for this single vertex.
        """
        sorted_eji_list = sorted(list(self._vertex_set), key=lambda eji: eji[1])
        renumbered_eji_list = [(j, eji[1]) for j, eji in enumerate(sorted_eji_list)]
        return Maximal_Simplex_Vertex(set(renumbered_eji_list))

    def check_valid(self):
        """Verify every j index appears at most once"""
        j_vals = {eji[0] for eji in self._vertex_set}
        assert len(j_vals) == len(self._vertex_set)

    def get_permuted_by(self, perm):
        """Apply permutation to j indices"""
        return Maximal_Simplex_Vertex({(perm[eji[0]], eji[1]) for eji in self._vertex_set})


@dataclass
class Eji_LinComb:
    """Represents a linear combination of elementary matrices Eji"""
    INT_TYPE = np.uint16  # uint16 should be enough for most cases

    _index: INT_TYPE
    _eji_counts: np.ndarray

    def __init__(self, n: int, k: int, list_of_Maximal_Simplex_Vertices: list[Maximal_Simplex_Vertex] | None = None):
        self._index = Eji_LinComb.INT_TYPE(0)
        self._eji_counts = np.zeros((n, k), dtype=Eji_LinComb.INT_TYPE, order='C')
        if list_of_Maximal_Simplex_Vertices:
            for msv in list_of_Maximal_Simplex_Vertices:
                self.add(msv)

    def index(self) -> INT_TYPE:
        """How many things were added together to make this Linear Combination."""
        return self._index

    def hash_to_point_in_unit_hypercube(self, dimension):
        """Hash the Eji_LinComb to a point in a unit hypercube of given dimension"""
        m = hashlib.md5()
        m.update(self._eji_counts)
        m.update(np.array([self._index]))
        ans = []
        for i in range(dimension):
            m.update(i.to_bytes(8))
            top_64_bits = int.from_bytes(m.digest()[:8], 'big')
            real_1 = np.float64(top_64_bits) / (1 << 64)
            ans.append(real_1)
        return np.asarray(ans)

    def add(self, msv: Maximal_Simplex_Vertex):
        """Add a Maximal_Simplex_Vertex to this linear combination"""
        self._index += 1
        for j, i in msv:
            self._eji_counts[j, i] += 1

    def __eq__(self, other: Self):
        return self._index == other._index and np.array_equal(self._eji_counts, other._eji_counts)

    def __ne__(self, other: Self):
        return not self.__eq__(other)

    def get_canonical_form(self) -> Self:
        """Get canonical form by sorting rows lexicographically"""
        ans = Eji_LinComb.__new__(Eji_LinComb)
        ans._index = self._index
        ans._eji_counts = sort_np_array_rows_lexicographically(self._eji_counts)
        return ans