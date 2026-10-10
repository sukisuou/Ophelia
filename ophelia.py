import json
import numpy as np
import tensorflow as tf

# set necessary data
from model import build_transformer, block_size

# build ophelia's model
print("Loading Ophelia...")
ophelia = build_transformer()
ophelia.load_weights("ophelia.weights.h5")
print("Done!\n")

# get her vocab dicts from `prepare.py`
with open("vocab.json", "r") as f:
    vocab = json.load(f)
char_to_id = vocab["char_to_id"]
id_to_char = {int(k): v for k, v in vocab["id_to_char"].items()}
# note: json keys are strings, cast to int

# create encoder and decoder
def encode(text):
    return [char_to_id[ch] for ch in text if ch in char_to_id]
def decode(ids):
    return "".join([id_to_char[i] for i in ids])

# create nucleus (top-p) temperature sampling - filter tail probabilities
def sample_token(logits, temperature = 0.7, top_p = 0.8):
    logits = logits / temperature

    # sort logits in descending order
    sorted_logits, sorted_indices = tf.math.top_k(logits, k = tf.shape(logits)[-1])

    # softmax and cumulative probabilities
    probs = tf.nn.softmax(sorted_logits, axis = -1)
    cum_probs = tf.math.cumsum(probs, axis = -1)

    # mask the tokens above the top_p threshold, probs > top_p
    mask = (cum_probs - probs) > top_p
    filtered_sorted_logits = tf.where(mask, float("-inf"), sorted_logits)

    # sample an index from the filtered candidates
    sampled_rank = tf.random.categorical(filtered_sorted_logits, num_samples = 1, dtype = tf.int32)

    # map back to vocab token ID
    next_token_id = tf.gather(sorted_indices[0], sampled_rank[0, 0])
    return tf.reshape(next_token_id, (1, 1))

# ----------------------------------------------------------------------------------------
# create autoregressive generation function
# ----------------------------------------------------------------------------------------
def generate(prompt, max_new_token = 60, stream = True):
    stop_id = char_to_id["|"]

    # encode prompt text to int32 ids (1, seq_len)
    tokens = encode(prompt)
    idx = tf.constant([tokens], dtype = tf.int32)

    # loop every token generation2
    for _ in range(max_new_token):
        # keep new tokens (limits to context window size, forgets old tokens)
        idx_cond = idx[:, -block_size:]

        # forward pass, grabs the last token (her output for this loop)
        logits = ophelia(idx_cond, training = False)
        last_token = logits[:, -1, :]

        # grab the next token (1, 1) using temperature of 0.7
        next_token_id = sample_token(last_token)

        # stops at delimiter "|"
        token_id = int(next_token_id[0, 0])
        if token_id == stop_id:
            break

        # stream letter by letter in real time
        if stream:
            print(id_to_char[token_id], end = "", flush = True)

        # concat back to idx (1, seq_len + 1)
        idx = tf.concat([idx, next_token_id], axis = 1)
    if stream:
        print('\n')

    # decode and return the generated text
    prompt_len = len(tokens)
    generated_tokens = idx[0, prompt_len:].numpy()
    return decode(generated_tokens)
# ----------------------------------------------------------------------------------------

# input prompting
if __name__ == "__main__":
    while(True):
        print("You: ", end = "", flush = True)
        prompt = input()
        if prompt.strip() == '/':
            print("Ophelia: Goodbye!")
            break
        print("Ophelia: ", end = "", flush = True)
        formatted_prompt = f"[{prompt}]"
        response = generate(formatted_prompt)