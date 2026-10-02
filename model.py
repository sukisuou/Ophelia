import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from dataset import data, block_size, get_batch

# 1. create a class for the embedding layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model = 64, block_size = block_size):
        super().__init__()

        # Token Embedding - lookup table (vocab_size, d_model)
        self.token_embed = layers.Embedding(input_dim = vocab_size, output_dim = d_model)

        # Positional Embedding (Learned Absolute)
        self.pos_embed = layers.Embedding(input_dim = block_size, output_dim = d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]     # grab only slider dim (block_size)
        positions = tf.range(start = 0, limit = seq_len)
        return self.token_embed(x) + self.pos_embed(positions)  # return input vector, X

# Testing
if __name__ == "__main__": 
    with open("vocab.json", "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    tokens, _ = get_batch(data) # (16, 64)

    embed_layer = TokenAndPositionEmbedding(vocab_size)
    X = embed_layer(tokens)     # (16, 64, 64)
    print(f"X shape: {X.shape}")