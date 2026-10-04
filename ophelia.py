import json
import numpy as np
import tensorflow as tf

# set necessary data
from model import build_transformer, block_size

# build ophelia's model
ophelia = build_transformer()
ophelia.load_weights("ophelia_weights.weights.h5")

# get her vocab dicts from `prepare.py`
with open("old_vocab.json", "r") as f:
    vocab = json.load(f)
char_to_id = vocab["char_to_id"]
id_to_char = {int(k): v for k, v in vocab["id_to_char"].items()}
# note: json keys are strings, cast to int

# create encoder and decoder
def encode(text):
    return [char_to_id[ch] for ch in text if ch in char_to_id]
def decode(ids):
    return "".join([id_to_char[i] for i in ids])

# create autoregressive generation function
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
        next_token_id = tf.random.categorical(last_token / 0.7, num_samples = 1, dtype = tf.int32)

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