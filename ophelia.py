import json
import numpy as np
import tensorflow as tf
from dataset import block_size
from model import build_transformer

# build ophelia's model
ophelia = build_transformer()
ophelia.load_weights("ophelia_weights.weights.h5")

# get her vocab dicts from `prepare.py`
with open("vocab.json", "r") as f:
    vocab = json.load(f)
char_to_id = vocab["char_to_id"]
id_to_char = {int(k): v for k, v in vocab["id_to_char"].items()}
# note: json keys are strings, cast to int

# create encoder and decoder
def encode(text):
    return [char_to_id[ch] for ch in text]
def decode(ids):
    return "".join([id_to_char[i] for i in ids])

# create autoregressive generation function
def generate(prompt, max_new_token = 60):
    stop_id = char_to_id["|"]

    # encode prompt text to int32 ids (1, seq_len)
    tokens = encode(prompt)
    idx = tf.constant([tokens], dtype = tf.int32)

    # loop every token generation
    for _ in range(max_new_token):
        # keep new tokens (limits to context window size, forgets old tokens)
        idx_cond = idx[:, -block_size:]

        # forward pass, grabs the last token (her output for this loop)
        logits = ophelia(idx_cond, training = False)
        last_token = logits[:, -1, :]

        # grab the highest probability index for next token (1,)
        next_token_id = tf.argmax(last_token, axis = -1, output_type = tf.int32)
        
        # expand to match 2D dim of idx (1, 1)
        next_token_id = tf.expand_dims(next_token_id, axis = -1)

        # stops at delimiter "|"
        if int(next_token_id[0, 0]) == stop_id:
            break

        # concat back to idx (1, seq_len + 1)
        idx = tf.concat([idx, next_token_id], axis = 1)
    
    # decode and return the generated text
    prompt_len = len(tokens)
    generated_tokens = idx[0, prompt_len:].numpy()
    return decode(generated_tokens)

# input prompting
if __name__ == "__main__":
    while(True):
        print("You: ", end = "", flush = True)
        prompt = input()
        print("Ophelia: ", end = "", flush = True)
        formatted_prompt = f"[{prompt}]"
        response = generate(formatted_prompt)
        print(response, '\n')