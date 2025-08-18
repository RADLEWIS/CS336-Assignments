import json
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def save_token_ids(token_ids: list, save_path: str):
    arr = np.array(token_ids, dtype=np.uint16)
    np.save(save_path, arr)
    print(f"Successfully save encoded text at {save_path}.")

train_dataset, vocab_size = "TinyStoriesV2-GPT4-", 10000
vocab_path, merges_path, special_tokens = f"data/{train_dataset}_{vocab_size}_vocab.json", f"data/{train_dataset}_{vocab_size}_merges.json", ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
with open(f"data/{train_dataset}valid.txt", "r", encoding="utf-8") as f:
    text = f.read()
text_bytes = text.encode("utf-8")
print(f"whole_bytes_len: {len(text_bytes)}")


sta_time = time.time()
text_encoded = tokenizer.encode(text)

end_time = time.time()
total_time = end_time - sta_time
print(f"in bytes per sec: {len(text_bytes) / total_time}")

compression_ratio = len(text_bytes) / len(text_encoded)
print(f"compression_ratio: ", compression_ratio)

save_path = f"data/{train_dataset}_token_ids.npy"
save_token_ids(text_encoded, save_path)