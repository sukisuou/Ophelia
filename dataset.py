# purpose: 
# - grabs dataset in batches
# - used in training

import numpy as np

# import data
data = np.load("encoded_data.npy")
print(f"Loaded {len(data)} tokens, shape: {data.shape}")

# next-token prediction
block_size = 32
def get_batch(data, batch_size = 16, block_size = block_size):
    max_start = len(data) - block_size - 1      # block size is slider size
    indices = np.random.randint(0, max_start, size = batch_size)

    # create 2D tensor with batch_size random blocks
    X = np.stack([data[i : i + block_size] for i in indices])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in indices])

    return X, y

if __name__ == "__main__": 
    X, y = get_batch(data)
    print(f"X: {X.shape}\n{X}\ny: {y.shape}\n{y}")