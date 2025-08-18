from numpy import typing as npt
from cs336_basics.transformer import *
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.optimizer import load_check_point

params = {
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512, 
    "num_heads": 16, 
    "num_layers": 4,
    "d_ff": 1344,
    "theta": 10000,
    "device": 'cpu', 
    "dtype": torch.float32,
}

def softmax_temp(x: torch.Tensor, i: int, temp: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x / temp
    max_entry = torch.max(x, dim=i, keepdim=True).values
    x_ = x - max_entry
    x_ = torch.exp(x_)
    sum_entry = torch.sum(x_, dim=i, keepdim=True)
    pred = x_ / sum_entry
    return pred.to(dtype)

def top_p(pred: torch.Tensor, p: float):
    """
    suppose pred is a single dim distribution
    return: pred
    """
    # 就地sort
    sorted = pred.sort(dim=-1, descending=True)
    indices = sorted.indices
    probs = sorted.values.detach()
    sum_p = 0.0
    v = -1
    for i, q in enumerate(probs):
        sum_p += q
        if sum_p + q < p:
            continue
        else:
            v = i
            break
    for i in range(pred.shape[-1]):
        if i <= v:
            pred[indices[i]] /= sum_p
        else:
            pred[indices[i]] = 0
    return pred


def decode(model_path, prompt: list[int], max_token: int, temp: float, p: float, special_token_ids: list[int]):
    """输入一条长度为n的prompt, 输出文本"""
    transformer_lm = Transformer_LM(**params)
    load_check_point(model_path, transformer_lm)
    output = prompt
    in_feature = torch.tensor(prompt, dtype=torch.long, device=params["device"])
    while in_feature.shape[-1] < max_token:
        if in_feature.shape[-1] > params["context_length"]:
            in_feature = in_feature[1:]
        with torch.no_grad():
            logits = transformer_lm(in_feature.unsqueeze(0)).squeeze(0)[-1]
            pred = softmax_temp(logits, -1, temp)
            pred = top_p(pred, p)
            sampled_idx = torch.multinomial(pred, num_samples=1)
            in_feature = torch.cat([in_feature, sampled_idx])
            output.append(sampled_idx)
            if sampled_idx in special_token_ids:
                break
    return output


vocab_path = "data/TinyStoriesV2-GPT4-_10000_vocab.json"
merge_path = "data/TinyStoriesV2-GPT4-_10000_merges.json"
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)

model_path = "checkpoints/checkpoint.pth"
prompt_str = "Once upon a time, there was a magical kingdom where a handsome prince lived."

prompt = tokenizer.encode(prompt_str)
special_token_ids = [tokenizer.encode(special_token)[0] for special_token in special_tokens]
output = decode(model_path, prompt, 1000, 1.0, 0.8, special_token_ids)
print(tokenizer.decode(output))