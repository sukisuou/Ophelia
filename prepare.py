# purpose: 
# - tokenization and data serialization
# - processes raw conversations from `convos`
# - saves the vocabulary and encoded data 

import json
import numpy as np
from convos import convos

# join into one stream
full_text = "".join(convos)
print(f"Total chars: {len(full_text)}")

# extract unique chars
chars = sorted(list(set(full_text)))
delimiters = ['_', '[', ']', '|']   # _: padding, []: prompt, |: response end
remaining_chars = [ch for ch in chars if ch not in delimiters]
vocab = delimiters + remaining_chars
print(f"Vocab: {vocab} ({len(vocab) - 4} unique chars + {len(delimiters)} delimiters)")

# create id encoder and char decoder
char_to_id = {ch: i for i, ch in enumerate(vocab)}
id_to_char = {i: ch for i, ch in enumerate(vocab)}

# tokenize
encoded_data = np.array([char_to_id[ch] for ch in full_text])
print(f"First 5 tokens: {encoded_data[:5]}")

# save data
vocab_data = {
    "char_to_id": char_to_id,
    "id_to_char": id_to_char
}

with open("vocab.json", "w", encoding = "utf-8") as f:
    json.dump(vocab_data, f, indent = 2, ensure_ascii = False)

np.save("encoded_data.npy", encoded_data)