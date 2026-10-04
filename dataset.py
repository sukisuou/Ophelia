# purpose: 
# - grabs dataset in batches
# - used in training

import numpy as np

# import data
data_pretrain = np.load("encoded_pretrain.npy")
data_persona = np.load("encoded_persona.npy")
if __name__ == "__main__": 
    print(f"Pretrain: Loaded {len(data_pretrain)} tokens, shape: {data_pretrain.shape}")
    print(f"Persona: Loaded {len(data_persona)} tokens, shape: {data_persona.shape}")

# next-token prediction
def get_batch(data, batch_size = 16, block_size = 64):
    max_start = len(data) - block_size - 1      # block size is slider size
    indices = np.random.randint(0, max_start, size = batch_size)

    # create 2D tensor with batch_size random blocks
    X = np.stack([data[i : i + block_size] for i in indices])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in indices])

    return X, y

# testing
if __name__ == "__main__": 
    X, y = get_batch(data_pretrain, batch_size = 2, block_size = 6)
    print(f"Example:\nX: {X.shape}\n{X}\ny: {y.shape}\n{y}")