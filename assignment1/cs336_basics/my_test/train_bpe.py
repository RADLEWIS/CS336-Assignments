import time
import json
from cs336_basics.run_bpe import *

def show_longest_token(vocabs: dict[int, bytes]):
    max_len = 0
    longest_vocab = None
    for id, vocab in vocabs.items():
        if len(vocab) > max_len:
            max_len = len(vocab)
            longest_vocab = vocab
    print(f"Longest token: {longest_vocab}")

def byte_to_str(b: bytes) -> str:
    s = ""
    for byte in b:
        s += chr(byte)
    return s

def save_bpe_model(vocabs: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    vocabs = {k: byte_to_str(v) for k, v in vocabs.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=True, indent=4)

    merges = [(byte_to_str(m[0]), byte_to_str(m[1])) for m in merges]
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False, indent=4)

start_time = time.time()
train_dataset = "TinyStoriesV2-GPT4-"
input_path, vocab_size, special_tokens = f"data/{train_dataset}valid.txt", 10000, ["<|endoftext|>"]
vocabs, merges = run_train_bpe_(input_path, vocab_size, special_tokens)
end_time = time.time()

total_time = end_time - start_time
print(f"Total time: {int(total_time / 60)}min {int(total_time - 60 * int(total_time / 60))}sec")

vocab_path = f"data/{train_dataset}_{vocab_size}_vocab.json"
merges_path = f"data/{train_dataset}_{vocab_size}_merges.json"
save_bpe_model(vocabs, merges, vocab_path, merges_path)


