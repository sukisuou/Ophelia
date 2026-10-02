import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# set necessary data
from dataset import data, block_size, get_batch
with open("vocab.json", "r") as f:
    vocab = json.load(f)
vocab_size = len(vocab["char_to_id"])
d_model = 64


# 1. create a class for the embedding layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model = d_model, block_size = block_size, **kwargs):
        super().__init__(**kwargs)

        # Token Embedding - lookup table (vocab_size, d_model)
        self.token_embed = layers.Embedding(input_dim = vocab_size, output_dim = d_model, name = "token_embedding")

        # Positional Embedding (Learned Absolute)
        self.pos_embed = layers.Embedding(input_dim = block_size, output_dim = d_model, name = "pos_embedding")

    def call(self, x):
        seq_len = block_size    # grab only slider dim (block_size)
        positions = tf.range(start = 0, limit = seq_len)
        return self.token_embed(x) + self.pos_embed(positions)  # return input vector, X


# 2. create multi-head attention (MHA) block
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model = d_model, num_heads = 4, **kwargs):
        super().__init__(**kwargs)

        # get h and d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # create projection dense layers (Q, K, V, W_0)
        self.q_proj = layers.Dense(d_model, use_bias = False, name = "Q_projection")
        self.k_proj = layers.Dense(d_model, use_bias = False, name = "K_projection")
        self.v_proj = layers.Dense(d_model, use_bias = False, name = "V_projection")
        self.out_proj = layers.Dense(d_model, use_bias = False, name = "W_0_projection")

    def split_heads(self, tensor, B, T):
        # (B, T, d_model) -> (B, T, h, d_k)
        tensor = tf.reshape(tensor, (B, T, self.num_heads, self.d_k))

        # (B, T, h, d_k) -> (B, h, T, d_k)
        return tf.transpose(tensor, perm = [0, 2, 1, 3])

    def call(self, x, mask = None):
        # get the projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # get batch and context length
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # split into num_heads heads (reshape and transpose)
        q = self.split_heads(q, B, T)
        k = self.split_heads(k, B, T)
        v = self.split_heads(v, B, T)

        # QK^T and scaling : (T, d_k) x (d_k, T) -> (T, T)
        att_scores = tf.matmul(q, k, transpose_b = True)
        att_scores /= tf.math.sqrt(tf.cast(self.d_k, att_scores.dtype))

        # causal masking
        if mask is not None:
            att_scores += mask

        # softmax
        A = tf.nn.softmax(att_scores, axis = -1)

        # att_out = AV : (T, T) x (T, d_k) -> (T, d_k) -> weight proj, W_0
        out = tf.matmul(A, v)
        out = tf.transpose(out, perm = [0, 2, 1, 3])    # (B, T, h, d_k)
        out = tf.reshape(out, (B, T, self.d_model))     # (B, T, d_model)
        return self.out_proj(out)


# transformer network with functional API
def build_transformer(block_size = block_size, d_model = d_model, num_heads = 4, num_stack = 4):
    inputs = layers.Input(shape = (block_size,), dtype = tf.int32, name = "token_ids")

    # 1. Token and Position embedding
    x = TokenAndPositionEmbedding(vocab_size)(inputs)

    # 2. Causal Mask
    rows = tf.range(block_size)[:, None]
    cols = tf.range(block_size)[None, :]
    causal_mask = tf.where(cols > rows, float('-inf'), 0.0)

    # 3. Transformer block
    for i in range(num_stack):
        # --- Attention sub-block (Pre-LN) ---
        norm_x = layers.LayerNormalization(epsilon = 1e-5, name = f"ln_att_{i}")(x)
        att_out = MultiHeadAttention(name = f"mha_{i}")(norm_x, mask = causal_mask)
        x = layers.Add(name = f"residual_att_{i}")([x, att_out])  # X = X + att(X)

        # --- Feed-Forward sub-block (Pre-LN) ---
        # next up

    # 4. Final Output Head
    norm_x = layers.LayerNormalization(epsilon = 1e-5, name = "ln_final")(x)
    logits = layers.Dense(vocab_size, name = "logits")(norm_x)   # linear output / logit

    return tf.keras.Model(inputs = inputs, outputs = logits, name = "DecoderTransformer")


# testing
if __name__ == "__main__":
    model = build_transformer()
    model.summary()