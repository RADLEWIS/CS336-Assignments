import copy
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from einops import rearrange
from typing import *
from cs336_alignment.math_baseline import r1_zero_reward_fn
from cs336_alignment.sft import tokenizer_prompt_and_output, get_reponse_log_probs
def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str,float]], 
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    # rollout_batch_size
    rewards = []
    return_dict = {}
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(response, ground_truth)
        rewards.append(reward["reward"])
    raw_rewards = torch.tensor(rewards)
    raw_rewards = rearrange(raw_rewards, "(n_prompts group_size) -> n_prompts group_size", group_size=group_size)
    mean_rewards = torch.mean(raw_rewards, dim=-1, keepdim=True)
    normalized_rewards = raw_rewards - mean_rewards
    return_dict["mean"] = mean_rewards

    if normalize_by_std:
        std_rewards = torch.std(raw_rewards, dim=-1, keepdim=True)
        normalized_rewards /= std_rewards + advantage_eps
        return_dict["std"] = std_rewards
    normalized_rewards = rearrange(normalized_rewards, "n_prompts group_size -> (n_prompts group_size)")
    raw_rewards = rearrange(raw_rewards, "n_prompts group_size -> (n_prompts group_size)")
    return normalized_rewards, raw_rewards, return_dict
    
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs:torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grop_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # this is log probs!
    probs_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_probs_ratio = torch.clip(probs_ratio, max=1+cliprange, min=1-cliprange)
    # element-wise minimum
    clipped_loss = -torch.minimum(advantages * probs_ratio, advantages * clipped_probs_ratio)
    return clipped_loss, {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline","reinforce_with_baseline","grpo_clip"],
    raw_rewards: torch.Tensor | None=None,
    advantages: torch.Tensor | None=None,
    old_log_probs: torch.Tensor | None=None,
    cliprange: float | None=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grop_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None=None) -> torch.Tensor:
    masked_tensor = tensor * mask
    element_num = torch.sum(mask, dim=dim)
    return torch.sum(masked_tensor, dim=dim) / element_num

def grpo_mircobatch_train_step(
    policy_log_probs: torch.Tensor, 
    response_mask: torch.Tensor, 
    gradient_accumulation: int, 
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
    raw_rewards: torch.Tensor | None=None, 
    advantages: torch.Tensor | None=None, 
    old_log_probs: torch.Tensor | None=None, 
    cliprange: float | None=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, _ = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    per_prompts_mean = masked_mean(loss, response_mask) / gradient_accumulation
    # backward only when scalar
    per_prompts_mean.backward(retain_graph=True)
    return per_prompts_mean.detach(), {}



hyper_params = {
    "n_grpo_steps": 200, 
    "learning_rate": 1e-5, 
    "advantage_eps": 1e-6, 
    "rollout_batch_size": 256, 
    "group_size": 8, 
    "sampling_temperature": 1.0,
    "sampling_max_tokens": 1024, 
    "sampling_min_tokens": 4,
    "epochs_per_rollout_batch": 1, 
    "train_batch_size": 256, 
    "gradient_accumulation_steps": 128, 
    "gpu_memory_utilization": 0.85,  
    "loss_type": "reinforce_with_baseline",
    "use_std_normalization": True,
}

def get_prompt(question: str) -> str:
    return "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \nUser: " \
                + question \
                + "\n Assistant: <think>"

def grpo_train_loop(
    n_grpo_steps: int, 
    learning_rate: float, 
    advantage_eps: float, 
    rollout_batch_size: int, 
    group_size: int, 
    sampling_temperature: float, 
    sampling_max_tokens: int, 
    sampling_min_tokens: int,
    epochs_per_rollout_batch: int, 
    train_batch_size: int, 
    gradient_accumulation_steps: int, 
    gpu_memory_utilization: float,  
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        ],
    use_std_normalization: bool,
):
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    model_name = "Qwen/Qwen2.5-Math-1.5B"
    policy = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    df = pd.read_parquet("data/countdown/0000.parquet")
    questions = df.head()["nums"].values
    prompts = []
    for q in questions:
        prompts.append(get_prompt(q))
    answers = df.head()["target"]
    repeated_prompts = [p for p in prompts for _ in range(group_size)]
    repeated_answers = [ans for ans in answers for _ in range(group_size)]
    sampling_params = SamplingParams(
        temperature=sampling_temperature, 
        top_p=1.0, 
        max_tokens=sampling_max_tokens, 
        min_tokens=sampling_min_tokens,
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )
    batch_nums = (len(prompts) - 1) // n_prompts_per_rollout_batch + 1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for step in range(n_grpo_steps):
        for batch in range(batch_nums):
            optimizer.zero_grad()
            # (n_prompts_per_rollout_batch, seq_len)
            for micro_batch in range(n_microbatches_per_rollout_batch):
                start_index = batch * rollout_batch_size + micro_batch * micro_train_batch_size
                prompts_micro_batch = repeated_prompts[start_index: start_index + micro_train_batch_size]
                answer_mirco_batch = repeated_answers[start_index: start_index + micro_train_batch_size]
                # we can derictly input strings
                outputs = policy.generate(inputs=prompts_micro_batch, sampling_params=sampling_params)
                responses = [o.outputs[0].text for o in outputs]
                old_log_probs = torch.tensor([o.outputs[0].logprobs for o in outputs])
                advantages, raw_rewards, _ = compute_group_normalized_rewards(
                    r1_zero_reward_fn, 
                    responses, 
                    repeated_ground_truths=answer_mirco_batch, 
                    group_size=group_size, 
                    advantage_eps=advantage_eps, 
                    normalize_by_std=use_std_normalization
                    )
                dict = tokenizer_prompt_and_output(prompts_micro_batch, responses, tokenizer)
                mask = dict["response_mask"]
                input_ids = dict["input_ids"]
                labels = dict["labels"]
                for train_step in range(epochs_per_rollout_batch):
                    # parallel token-wise
                    model = policy.llm_engine.model
                    new_log_probs = get_reponse_log_probs(model, input_ids, labels, False)["log_probs"]
                    loss, _ = grpo_mircobatch_train_step(
                        new_log_probs, 
                        mask, 
                        gradient_accumulation_steps, 
                        loss_type, 
                        raw_rewards, 
                        advantages, 
                        old_log_probs, 
                        cliprange=0.2
                        )
                    print(f"Step: {step}, Train step: {train_step}, Loss: {loss}")
            optimizer.step()

if __name__ == "__main__":
    grpo_train_loop(**hyper_params)
