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

# use AdamW and sparse categorical CE loss 
model.compile(
    optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-3, weight_decay = 0.01, clipnorm = 1.0),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)

# stop early to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'loss',
    patience = 5,
    restore_best_weights = True,
    verbose = True
)

# begin training
history = model.fit(
    data_gen(),
    steps_per_epoch = 128,
    epochs = 40
)
np.save("training_history.npy", history.history)

# save weights to rebuild later
model.save_weights("ophelia.weights.h5")
print("Weights saved successfully!")