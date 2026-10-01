import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from dataset import data, get_batch

with open("vocab.json", "r") as f:
    vocab = json.load(f)
vocab_size = len(vocab)


# 1. Token Embedding - lookup table (vocab_size, d_model)
d_model = 64
token_embedding_layer = layers.Embedding(input_dim = vocab_size, output_dim = d_model)

# 2. Positional Embedding (Learned Absolute)
block_size = 8
pos_embedding_layer = layers.Embedding(input_dim = block_size, output_dim = d_model)



# Testing
dummy_tokens, _ = get_batch(data) # (4, 8)

# token embedding
E = token_embedding_layer(dummy_tokens)
print(f"E shape: {E.shape}")

# positional embedding
seq_len = tf.shape(dummy_tokens)[1]     # grab only slider dim (block_size)
positions = tf.range(start = 0, limit = seq_len)
P = pos_embedding_layer(positions)
print(f"P shape: {P.shape}")

# get input vector, X
X = E + P
print(f"X shape: {X.shape}")