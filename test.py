import tensorflow as tf

rows = tf.range(3)[:, None]
cols = tf.range(3)[None, :]

causal_mask = tf.where(cols > rows, float('-inf'), 0.0)
print(causal_mask)