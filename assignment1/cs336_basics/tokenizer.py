import os
import json
import regex as re
from typing import Iterable
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


def pre_tokens_iter(
    input_path: str | os.PathLike, 
    special_tokens: list[str]
):  
    PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
            pre_chunks = re.split("|".join(map(re.escape, special_tokens)), chunk)
            for pre_chunk in pre_chunks:
                for match in re.finditer(PAT, pre_chunk):
                    yield match.group() # 直接输出 token

def run_train_bpe_(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    init_vocab = {i: bytes([i]) for i in range(256)}
    i_init_vocab = {bytes([i]): i for i in range(256)}
    cnt = 256
    freq_table = []
    # 通过word对应的token pair访问word, word_idx
    word_map = {}
    i_word_map = {}
    word_cnt = 0

    test_record = []
    for token in pre_tokens_iter(input_path=input_path, special_tokens=special_tokens):
        token_list_key = []
        token_bytes = token.encode("utf-8")
        for b in token_bytes:
            byte_token = bytes([b]) 
            if byte_token not in i_init_vocab:
                init_vocab[cnt] = byte_token
                i_init_vocab[byte_token] = cnt
                cnt += 1
            token_list_key.append(i_init_vocab[byte_token])
        token_tuple_key = tuple(token_list_key)
        
        if token_tuple_key not in i_word_map:
            freq_table.append([1, token_tuple_key])
            word_map[word_cnt] = token_tuple_key
            i_word_map[token_tuple_key] = word_cnt
            word_cnt += 1
        else:
            freq_table[i_word_map[token_tuple_key]][0] += 1

    merges = []
    pair_table = defaultdict(int)
    pair_to_word = defaultdict(dict)
    max_pair = (-1, -1)
    updated_word_idxes = []
    old_word_map = {}
    while(len(init_vocab) < vocab_size - 1):
        if len(pair_table) == 0:
            for word_idx, freq_list in enumerate(freq_table):
                freq = freq_list[0]
                key = freq_list[1]

                i = 0
                while i < len(key) - 1:
                    pair = (key[i], key[i+1])
                    pair_table[pair] += freq
                    if word_idx in pair_to_word[pair]:
                        pair_to_word[pair][word_idx] += 1
                    else:
                        pair_to_word[pair][word_idx] = 1
                    i += 1
        else:
            pair_table.pop(max_pair)
            pair_to_word.pop(max_pair)

            for word_idx in updated_word_idxes:
                key = old_word_map[word_idx]
                freq = freq_table[word_idx][0]
                i = 0
                while i < len(key) - 1:
                    pair_ = (key[i], key[i+1])
                    if pair_ == max_pair:
                        id = cnt - 1
                        if i > 0:
                            pair = (key[i-1], key[i])

                            if pair in pair_table:
                                pair_table[pair] -= freq
                                if pair_table[pair] > 0:
                                    if word_idx in pair_to_word[pair]:
                                        pair_to_word[pair][word_idx] -= 1
                                        if pair_to_word[pair][word_idx] <= 0:
                                            pair_to_word[pair].pop(word_idx)
                                else:
                                    pair_table.pop(pair)
                                    pair_to_word.pop(pair)
                                
                            
                            new_pair = (key[i-1], id)
                            pair_table[new_pair] += freq
                            if word_idx in pair_to_word[new_pair]:
                                pair_to_word[new_pair][word_idx] += 1
                            else:
                                pair_to_word[new_pair][word_idx] = 1

                        if i < len(key) - 2:
                            pair = (key[i+1], key[i+2])
                            
                            if pair in pair_table:
                                pair_table[pair] -= freq
                                if pair_table[pair] > 0:
                                    if word_idx in pair_to_word[pair]:
                                        pair_to_word[pair][word_idx] -= 1
                                        if pair_to_word[pair][word_idx] <= 0:
                                            pair_to_word[pair].pop(word_idx)
                                else:
                                    pair_table.pop(pair)
                                    pair_to_word.pop(pair)

                            new_pair = (id, key[i+2])
                            pair_table[new_pair] += freq
                            if word_idx in pair_to_word[new_pair]:
                                pair_to_word[new_pair][word_idx] += 1
                            else:
                                pair_to_word[new_pair][word_idx] = 1
                        i += 2
                    else:
                        i += 1


        max_pair = max(pair_table, key=lambda k: (pair_table[k], init_vocab[k[0]], init_vocab[k[1]]))
        test_record.append((pair_table[max_pair], len(pair_to_word[max_pair])))
        init_vocab[cnt] = init_vocab[max_pair[0]] + init_vocab[max_pair[1]]
        merges.append((init_vocab[max_pair[0]], init_vocab[max_pair[1]]))

        updated_word_idxes = pair_to_word[max_pair].keys()
        for word_idx in updated_word_idxes:
            # 只保存受影响的词的旧状态
            old_word_map[word_idx] = word_map[word_idx]

        for word_idx in updated_word_idxes:
            key = word_map[word_idx]
            new_key = []
            i = 0
            while i < len(key):
                if(key[i:i+len(max_pair)] == max_pair):
                    new_key.append(cnt)
                    i += len(max_pair)
                else:
                    new_key.append(key[i])
                    i += 1
            new_key = tuple(new_key)
            
            word_map[word_idx] = new_key
            freq_table[word_idx][1] = new_key

        cnt += 1
    for token in special_tokens:
        init_vocab[cnt] = token.encode("utf-8")
        cnt += 1
    # print(merges[:10], test_record[:10], [init_vocab[i] for i in range(256, 300)])
    return init_vocab, merges



class Tokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.i_vocab = {vocab[i]: i for i in vocab.keys()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
            cls, 
            vocab_filepath: str, 
            merges_filepath: str, 
            special_tokens: list[str] | None=None):
        """从文件中读取并构建Tokenizer. 须注意文件类型是json"""
        def str_encode(s: str) -> bytes:
            b = bytes()
            for c in s:
                if ord(c) >= 128 and ord(c) < 0xFF:
                    b += c.encode('latin-1')
                else:
                    b += c.encode('utf-8')
            return b
        with open(vocab_filepath, 'r') as f:
            vocab_str = json.load(f)
            vocab = {int(k): str_encode(v) for k, v in vocab_str.items()}

        with open(merges_filepath, 'r') as f:
            merges_str = json.load(f)
            merges = [(str_encode(m[0]), str_encode(m[1])) for m in merges_str]
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

        # step 1 按照special tokens分块
        chunks = []
        tokens_id_between = []
        tokens_pos = []
        total_cnt = 0
        if self.special_tokens is None:
            chunks.append(text)
        else:

            for special_token in self.special_tokens:
                token_pos = []
                sta = 0
                while True:
                    find_at = text.find(special_token, sta)
                    if find_at != -1:
                        token_pos.append(find_at)
                        sta = find_at + len(special_token)
                        total_cnt += 1
                    else:
                        break
                tokens_pos.append(token_pos)
            
            text_sta = 0
            while total_cnt > 0:
                first_find = 1e18
                token_id = -1
                for id, token_pos in enumerate(tokens_pos):
                    while len(token_pos) > 0 and token_pos[0] < text_sta:
                        token_pos.pop(0)
                        total_cnt -= 1
                    if len(token_pos) > 0:
                        if (
                            token_pos[0] < first_find
                            or (token_pos[0] == first_find and len(self.special_tokens[token_id]) < len(self.special_tokens[id]))
                            ):
                            first_find = token_pos[0]
                            token_id = id
                if token_id != -1:
                    chunks.append(text[text_sta:first_find])
                    tokens_id_between.append(token_id)
                    text_sta = first_find + len(self.special_tokens[token_id])
                    
                    total_cnt -= 1
                    tokens_pos[token_id].pop(0)
            chunks.append(text[text_sta:])

        # step 2 对每一块执行regex
        # step 3 对每一个word进行拆分
        # step 4 merge 循环 + 查询
        token_ids = []
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        for idx, chunk in enumerate(chunks):
            words = re.findall(PAT, chunk)
            for token in words:
                token_bytes = token.encode("utf-8")
                token_bytes_list = [bytes([b]) for b in token_bytes]
                for merge in self.merges:
                    i = 0
                    while True:
                        if i >= len(token_bytes_list) - 1:
                            break
                        pair = (token_bytes_list[i], token_bytes_list[i+1])
                        if pair == merge:
                            merged_token = token_bytes_list[i] + token_bytes_list[i+1]
                            token_bytes_list[i:i+2] = [merged_token]
                        else:
                            i += 1
                token_id = [self.i_vocab[t] for t in token_bytes_list]
                token_ids += token_id

            # 不确定最后一个chunk是否有special token
            if idx != len(tokens_id_between):
                special_token = self.special_tokens[tokens_id_between[idx]]
                special_token_id = self.i_vocab[special_token.encode("utf-8")]
                token_ids.append(special_token_id)
                            
        return token_ids



    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """给一个iterable, 慢慢yield token IDs."""
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        """将一串token IDs decode 成text."""
        text = bytes([])
        for id in ids:
            token = self.vocab[id]
            """
            if len(token) >= 2:
                print(token)"""
            text += token
        return text.decode("utf-8", errors="replace")
