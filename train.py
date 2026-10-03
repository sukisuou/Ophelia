import numpy as np
import tensorflow as tf

# set necessary data
from model import build_transformer, d_model
from dataset import data, block_size, get_batch

# split data (90/10)
# too small for her!!! split on bigger one later

# create a dataset generator with get_batch
def data_gen(batch_size = 16):
    while True:
        yield get_batch(data, batch_size = batch_size)

# build the model
model = build_transformer()
model.summary()

# use Adam and sparse categorical CE loss 
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)

# begin training
history = model.fit(
    data_gen(),
    steps_per_epoch = 64,
    epochs = 20
)
np.save("training_history.npy", history.history)

# save weights to rebuild later
model.save_weights("ophelia_weights.weights.h5")
print("Weights saved successfully!")