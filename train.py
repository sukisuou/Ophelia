import numpy as np
import tensorflow as tf

# set necessary data
from dataset import data_persona, get_batch
from model import build_transformer, block_size

# create a dataset generator with get_batch
def data_gen(batch_size = 16, block_size = block_size):
    while True:
        yield get_batch(data_persona, batch_size = batch_size, block_size = block_size)

# build the model
model = build_transformer()
model.load_weights("ophelia_pretrain.weights.h5")
model.summary()

# use Adam and sparse categorical CE loss 
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)

# begin training
history = model.fit(
    data_gen(),
    steps_per_epoch = 32,
    epochs = 20
)
np.save("training_history.npy", history.history)

# save weights to rebuild later
model.save_weights("ophelia.weights.h5")
print("Weights saved successfully!")