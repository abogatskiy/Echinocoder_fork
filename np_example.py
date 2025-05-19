#!/usr/bin/env python3

import numpy as np
import time

b = 2
num_regular = 3
num_zeros = 2
n = num_regular + num_zeros
k_base = 3  # number of non-mask features
k = k_base + 1  # last component is the mask
np.random.seed(0)

# Create regular vectors
regular_vectors = np.random.rand(b, num_regular, k_base)
# Add mask=1 to regular vectors
regular_vectors = np.concatenate([np.ones((b, num_regular, 1)), regular_vectors], axis=2)

# Create zero vectors
zero_vectors = np.zeros((b, num_zeros, k_base))
# Add mask=0 to zero vectors
zero_vectors = np.concatenate([np.zeros((b, num_zeros, 1)), zero_vectors], axis=2)

# Concatenate regular and zero vectors
set_of_vectors_to_embed = np.concatenate([regular_vectors, zero_vectors], axis=1)

arr = set_of_vectors_to_embed

import simplex1
import simplex2
from np_dotting_embedder import DottingEmbedder

np.random.seed(0)
d2 = DottingEmbedder(n, k)

some_embedders = [
    simplex1.Embedder(),
    simplex2.Embedder(),
    d2
]


print("")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("")

print(f"\n==========================\nVectors to be encoded are:|")
for vec in arr:
    print(f"{vec}, ") 

concat = [n, k]

for embedder in some_embedders:
    # Time the embedding process
    start_time = time.time()
    embedding, (n_out, k_out), metadata = embedder.embed(arr)
    elapsed_ms = (time.time() - start_time) * 1000
    if isinstance(embedder, DottingEmbedder):
        embedding = embedding.reshape(b, -1, n)

    assert (n_out, k_out) == (n, k)
    print(f"\nembedding for embedder {embedder} is \n{embedding}\n")
    print(f"Time taken: {elapsed_ms:.3f} ms")
    
    # Permutation test - apply one permutation to the entire batch
    print("=== Permutation Test ===")
    # Generate a random permutation
    perm = np.random.permutation(n)
    
    # Create permuted version of the entire batch
    permuted_arr = arr.copy()
    for i in range(b):
        permuted_arr[i] = permuted_arr[i][perm]
    
    # Embed the permuted batch
    perm_embedding, _, _ = embedder.embed(permuted_arr)
    if isinstance(embedder, DottingEmbedder):
        perm_embedding = perm_embedding.reshape(b, -1, n)
    
    # Calculate the max absolute difference
    max_diff = np.max(np.abs(embedding - perm_embedding))
    print(f"Max absolute difference after permutation: {max_diff}")
    print(f"PASS: Embedder is permutation invariant" if max_diff < 1e-10 else f"FAIL: Embedding changes after permutation (diff: {max_diff})")
    
    # Batch splitting test
    if b > 1:
        print("\n=== Batch Splitting Test ===")
        # Split the batch into two halves
        first_half = arr[:b//2]
        second_half = arr[b//2:]
        
        # Embed each half separately
        embedding_first_half, _, _ = embedder.embed(first_half)
        embedding_second_half, _, _ = embedder.embed(second_half)
        
        # Concatenate the results
        split_embedding = np.vstack((embedding_first_half, embedding_second_half))
        if isinstance(embedder, DottingEmbedder):
            split_embedding = split_embedding.reshape(b, -1, n)

        # Compare with the original embedding
        max_diff_batch = np.max(np.abs(embedding - split_embedding))
        print(f"Max absolute difference after batch splitting: {max_diff_batch}")
        print(f"PASS: Batch processing is consistent" if max_diff_batch < 1e-10 else f"FAIL: Batch processing is inconsistent (diff: {max_diff_batch})")
        
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")