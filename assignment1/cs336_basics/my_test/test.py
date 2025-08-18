from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab = tokenizer.get_vocab()

flag = True
# 查找包含 \x80 的 token
for tok, idx in vocab.items():
    if any(ord(c) >= 128 for c in tok):  # 非 ASCII
        print(repr(tok))
        flag = False

print(flag)
