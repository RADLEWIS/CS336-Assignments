import os
import math
import random
import numpy as np
from numpy import typing as npt
import torch

from einops import rearrange, einsum, reduce
from collections.abc import Callable, Iterable
from typing import Optional

def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.float32:
    dtype = logits.dtype
    logits = logits.to(torch.float32)
    # 记得keepdim, 否则不能用broadcast
    max_entry = torch.max(logits, dim=-1, keepdim=True).values
    l = torch.log(torch.sum(torch.exp(logits - max_entry), dim=-1, keepdim=True))
    l -= (logits.gather(dim=-1, index=target.unsqueeze(-1)) - max_entry)
    return l.mean().to(dtype)

class AdamW(torch.optim.Optimizer):
    def __init__(
            self, 
            params: torch.Tensor, 
            lr: float,
            weight_decay: float,
            betas: tuple[float, float],
            eps: float
            ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = 0.0
                state["v"] = 0.0

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # 注意这里的t从0开始, 而我们需要t从1开始
                t = state.get("t", 0)
                grad = p.grad.data
                # 字典索引拿到的不可变元素是拷贝(值本身), 需要手动修改
                state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * grad
                state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * (grad ** 2)
                m = state["m"]
                v = state["v"]
                lr_t = lr * math.sqrt(1.0 - self.beta2 ** (t + 1)) / (1.0 - self.beta1 ** (t + 1))
                p.data -= lr_t * m / (torch.sqrt(v) + self.eps)
                p.data -= lr * self.weight_decay * p.data
                state["t"] = t + 1
        return loss
    

    
def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t/T_w * lr_max
    elif T_w <= t <= T_c:
        return lr_min + 0.5 * (1 + math.cos((t - T_w)/(T_c - T_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    eps = 1e-6
    grads = []
    for param in params:
        g = param.grad
        if g is None:
            continue
        grads.append(g)
    # 此时创建了新张量, grads由param.grad复制而来
    # 维度不一定能对齐, 应该先展平
    grads = torch.cat([g.flatten() for g in grads])
    grads_l2_norm = torch.norm(grads)

    if grads_l2_norm > max_norm:
        for param in params:
            g = param.grad
            if g is None:
                continue
            g *= max_norm / (grads_l2_norm + eps)


def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: torch.device, dtype: torch.dtype=torch.int32) -> torch.Tensor:
    assert len(x) >= batch_size + context_length, "length of x is not enough"
    # starting points indices, random sampling
    indices = random.sample(range(0, len(x) - context_length), batch_size)
    features = torch.stack([torch.tensor(x[i:i+context_length]) for i in indices]).to(device=device, dtype=dtype)
    labels = torch.stack([torch.tensor(x[i+1:i+context_length+1]) for i in indices]).to(device=device, dtype=dtype)
    return features, labels

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    obj = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)

def load_check_point(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None=None):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer_state"])
    iteration = obj["iteration"]
    return iteration

def monitor_norm(params: Iterable[torch.nn.Parameter]):
    """Return tuple[params_lr_norm, grads_lr_norm]"""
    grads_ = []
    params_ = []
    for param in params:
        params_.append(param.data)
        g = param.grad
        if g is None:
            continue
        grads_.append(g)

    params = torch.cat([p.flatten() for p in params_])
    params_l2_norm = torch.norm(params)
    grads = torch.cat([g.flatten() for g in grads_])
    grads_l2_norm = torch.norm(grads)

    return params_l2_norm, grads_l2_norm
