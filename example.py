#!/usr/bin/env python3

import numpy as np

set_of_vectors_to_embed = np.array([[
      (8,-1,-4,3),
      (-8,-5,9,7),
      (8,2,7,-7)]])
another_set_of_vectors_to_embed = np.array([[
      (8,2,7,-7),
      (8,-1,-4,3),
      (-8,-5,9,7)]])
tweaked_set_of_vectors_to_embed = np.array([[
      (8,-1,-4,3),
      (-8,-5,9,7),
      (8,2,7,-7.001)]])

b,n,k = set_of_vectors_to_embed.shape

test_sets_of_vectors = [ 
        set_of_vectors_to_embed,
        another_set_of_vectors_to_embed,
        tweaked_set_of_vectors_to_embed, 
    ]

import simplex1
import simplex2
import dotting

some_embedders = [ 
    simplex1.Embedder(),
    simplex2.Embedder(),
    dotting.Embedder(n,k)
]


for arr in test_sets_of_vectors:
    print("")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("")
    
    print(f"\n==========================\nVectors to be encoded are:|")
    for vec in arr:
       print(f"{vec}, ") 

    concat = [ n, k]
    for embedder in some_embedders:
        embedding, (n_out, k_out), metadata = embedder.embed(arr)
        assert (n_out, k_out) == (n, k)
        print(f"\nembedding for embedder {embedder} is \n{embedding}\n")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
