# pre-train with tinystories dataset for language learning

import numpy as np
import tensorflow as tf

# set necessary data
from dataset import data_pretrain, get_batch
from model import build_transformer, block_size

# split data (90/10)
split_idx = int(len(data_pretrain) * 0.9)
train_data = data_pretrain[:split_idx]
val_data = data_pretrain[split_idx:]

# create a dataset generator with get_batch
def data_gen(data, batch_size = 16, block_size = block_size):
    while True:
        yield get_batch(data, batch_size = batch_size, block_size = block_size)

# build the model
model = build_transformer()
model.summary()

# use AdamW and sparse categorical CE loss 
model.compile(
    optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-3, weight_decay = 0.01, clipnorm = 1.0),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)

# stop early to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True,
    verbose = True
)

# begin training
history = model.fit(
    data_gen(train_data),
    steps_per_epoch = 256,
    epochs = 40,
    validation_data = data_gen(val_data),
    validation_steps = 16,
    callbacks = [early_stopping]
)
np.save("pretrain_history.npy", history.history)

# save weights to rebuild later
model.save_weights("ophelia_pretrain.weights.h5")
print("Pretrained weights saved successfully!")