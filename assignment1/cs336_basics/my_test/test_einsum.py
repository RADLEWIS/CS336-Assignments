#  uv run --active python cs336_basics/test/test_einsum.py
import torch
from einops import einsum, rearrange

a = torch.randn([5, 32, 16])
b = torch.randn([5, 32, 16])
c = einsum(a, b, "... h d_k, ... w d_k -> ... h w")