# purpose: 
# - tokenization and data serialization
# - processes raw conversations from `convos` and 'tinystories_dataset.txt`
# - saves the vocabulary and encoded data 

import json
import string
import numpy as np
from convos import convos

# grab the datasets (pretrain and persona)
persona_text = "".join(convos)
with open("tinystories_dataset.txt", "r", encoding = "utf-8") as f:
    ts_text = f.read()
ts_text = ( # optional cleanup
    ts_text.replace('“', '"')
           .replace('”', '"')
           .replace('’', "'")
           .replace('‘', "'")
)
flat_ts_text = " ".join(ts_text.split())

# filter out non-ascii garbage chars
delimiters = ['_', '[', ']', '|']   # _: padding, []: prompt, |: response end
allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " " + "~") - {'\\'}
clean_ts_text = "".join([ch for ch in flat_ts_text if ch in allowed_chars or ch in delimiters])

# join into one stream
full_text = clean_ts_text + " " + persona_text
print(f"TinyStories chars   : {len(clean_ts_text)}")
print(f"Personal chars      : {len(persona_text)}")
print(f"Total chars         : {len(full_text)}")

# extract unique chars
chars = sorted(list(set(full_text)))
remaining_chars = [ch for ch in chars if ch not in delimiters]
vocab = delimiters + remaining_chars
print(f"Vocab: {vocab} ({len(vocab)} unique chars - including {len(delimiters)} delimiters)")

# create id encoder and char decoder
char_to_id = {ch: i for i, ch in enumerate(vocab)}
id_to_char = {i: ch for i, ch in enumerate(vocab)}

# tokenize
encoded_pretrain = np.array([char_to_id[ch] for ch in clean_ts_text], dtype = np.int32)
encoded_persona = np.array([char_to_id[ch] for ch in persona_text], dtype = np.int32)
print(f"First 5 pretrain tokens: {encoded_pretrain[:5]}")
print(f"First 5 persona tokens: {encoded_persona[:5]}")

# save data
vocab_data = {
    "char_to_id": char_to_id,
    "id_to_char": id_to_char
}

with open("vocab.json", "w", encoding = "utf-8") as f:
    json.dump(vocab_data, f, indent = 2, ensure_ascii = False)

np.save("encoded_pretrain.npy", encoded_pretrain)
np.save("encoded_persona.npy", encoded_persona)

print("Saved vocab.json, encoded_pretrain, and encoded_persona!")