#!/usr/bin/env python3

import torch
import time

# Set random seed for reproducibility
torch.manual_seed(0)

# Test parameters
b = 1000  # batch size
num_regular = 80
num_zeros = 2
n = num_regular + num_zeros
k_base = 3  # number of non-mask features
k = k_base + 1  # last component is the mask

# Create regular vectors
regular_vectors = torch.rand(b, num_regular, k_base, dtype=torch.float64)
# Add mask=1 to regular vectors
regular_vectors = torch.cat([torch.ones(b, num_regular, 1, dtype=torch.float64), regular_vectors], dim=2)

# Create zero vectors
zero_vectors = torch.zeros(b, num_zeros, k_base, dtype=torch.float64)
# Add mask=0 to zero vectors
zero_vectors = torch.cat([torch.zeros(b, num_zeros, 1, dtype=torch.float64), zero_vectors], dim=2)

# Concatenate regular and zero vectors
set_of_vectors_to_embed = torch.cat([regular_vectors, zero_vectors], dim=1)

arr = set_of_vectors_to_embed

# Import and initialize the embedder
from torch_dotting_embedder import TorchDottingEmbedder

torch.manual_seed(0)
embedder = TorchDottingEmbedder(n, k, device=torch.device('cpu'), dtype=torch.float64)

print("")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("")

print(f"\n==========================\nVectors to be encoded are:|")
for vec in arr:
    print(f"{vec}, ")

# Time the embedding process
start_time = time.time()
embedding, (n_out, k_out), metadata = embedder(arr)
elapsed_ms = (time.time() - start_time) * 1000
embedding = embedding.reshape(b, -1, n)

assert (n_out, k_out) == (n, k)
print(f"\nembedding for embedder {embedder} is \n{embedding}\n")
print(f"Time taken: {elapsed_ms:.3f} ms")

# Permutation test - apply one permutation to the entire batch
print("=== Permutation Test ===")
# Generate a random permutation
perm = torch.randperm(n)

# Create permuted version of the entire batch
permuted_arr = arr.clone()
for i in range(b):
    permuted_arr[i] = permuted_arr[i][perm]

# Embed the permuted batch
perm_embedding, _, _ = embedder(permuted_arr)
perm_embedding = perm_embedding.reshape(b, -1, n)

# Calculate the max absolute difference
max_diff = torch.max(torch.abs(embedding - perm_embedding)).item()
print(f"Max absolute difference after permutation: {max_diff}")
print(f"PASS: Embedder is permutation invariant" if max_diff < 1e-10 else f"FAIL: Embedding changes after permutation (diff: {max_diff})")

# Batch splitting test
if b > 1:
    print("\n=== Batch Splitting Test ===")
    # Split the batch into two halves
    first_half = arr[:b//2]
    second_half = arr[b//2:]
    
    # Embed each half separately
    embedding_first_half, _, _ = embedder(first_half)
    embedding_second_half, _, _ = embedder(second_half)
    
    # Concatenate the results
    split_embedding = torch.cat((embedding_first_half, embedding_second_half), dim=0)
    split_embedding = split_embedding.reshape(b, -1, n)

    # Compare with the original embedding
    max_diff_batch = torch.max(torch.abs(embedding - split_embedding)).item()
    print(f"Max absolute difference after batch splitting: {max_diff_batch}")
    print(f"PASS: Batch processing is consistent" if max_diff_batch < 1e-10 else f"FAIL: Batch processing is inconsistent (diff: {max_diff_batch})")
    
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") 