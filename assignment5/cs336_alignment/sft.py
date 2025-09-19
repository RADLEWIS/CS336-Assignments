import torch
# from einops import rearrange
from transformers import PreTrainedTokenizerBase, PreTrainedModel

# used for calculate entropies / importance sampling
# put prompt and output together & output input_ids and labels
def tokenizer_prompt_and_output(
    prompt_strs, output_strs, 
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    prompt_and_output_lens = []
    non_padded_tokens = []
    input_ids = []
    labels = []
    response_masks = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt_str)["input_ids"]
        output_tokens = tokenizer(output_str)["input_ids"]
        prompt_and_output_lens.append((len(prompt_tokens), len(output_tokens)))
        tokens = torch.tensor(prompt_tokens + output_tokens, dtype=torch.long)
        non_padded_tokens.append(tokens)
    max_len = max([a + b for a, b in prompt_and_output_lens])
    for i, non_padded_token in enumerate(non_padded_tokens):
        prompt_len = prompt_and_output_lens[i][0]
        output_len = prompt_and_output_lens[i][1]
        ans_len = prompt_len + output_len
        padding_len = max_len - ans_len

        padding_token = torch.full([padding_len], tokenizer.pad_token_id, dtype=torch.long)
        padded_token = torch.cat((non_padded_token, padding_token))

        response_masks.append(
            torch.cat(
                [torch.zeros([prompt_len-1], dtype=torch.long), 
                 torch.ones([output_len], dtype=torch.long), 
                 torch.zeros([padding_len], dtype=torch.long)
                 ]
                 ))
        input_ids.append(padded_token[:-1])
        labels.append(padded_token[1:])

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    response_masks = torch.stack(response_masks)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_masks
    }
    
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # make sure to keep dims
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    entropy = -1 * torch.sum(probs * log_probs, dim=-1)
    return entropy
    

@torch.no_grad()
def get_reponse_log_probs(
    model: PreTrainedModel, 
    input_ids: torch.Tensor, 
    labels: torch.Tensor, 
    return_token_entropy: bool=False
) -> dict[str, torch.Tensor]:
    dict = {}
    logits = model(input_ids).logits
    # we need torch.gather to access corrd log_prob
    log_probs = torch.gather((
        logits - torch.logsumexp(logits, dim=-1, keepdim=True)), 
        dim=-1,
        index=labels.unsqueeze(-1)
        ).squeeze(-1)
    dict["log_probs"] = log_probs
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        dict["token_entropy"] = token_entropy
    return dict

# calculate a tensor(usually for loss per token)
def masked_normalize(
    tensor: torch.Tensor, 
    mask: torch.Tensor, 
    normalize_constant: float, 
    dim: int | None = None,
) -> torch.Tensor:
    '''maxmize the log likelihood of the target output'''
    masked_tensor = tensor * mask
    # (batch_size,)
    summed_tensor = torch.sum(masked_tensor, dim=dim) / normalize_constant
    return summed_tensor

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor, 
    response_mask: torch.Tensor, 
    gradient_accumulation_steps: int, 
    normalize_constant: float=1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masked_log_sum = masked_normalize(policy_log_probs, response_mask, normalize_constant, -1)
    loss = -torch.mean(masked_log_sum) / gradient_accumulation_steps
    # keep graph cause after this func will be called again
    loss.backward(retain_graph=True)
    loss_detach = loss.detach()
    return loss_detach, {"loss": loss_detach}
