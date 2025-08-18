import torch
from einops import rearrange, einsum
from jaxtyping import Float, Int
class Linear(torch.nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
            ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        # shape: d_out, d_in
        self.weight = torch.nn.Parameter(torch.zeros([self.out_features, self.in_features], device=device, dtype=dtype), requires_grad=True)
        sigma = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x.to(self.weight.device)
        out = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return out

class Embedding(torch.nn.Module):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
            ):
        """
        num_embeddings: size of the vocabulary.
        embedding_dim: dim of the embedding vectors. i.e., d_model.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.zeros([num_embeddings, embedding_dim], dtype=self.dtype, device=self.device), requires_grad=True)
        sigma = (2 / (num_embeddings + embedding_dim)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.weight.device)
        out = self.weight[token_ids]
        # 不能return torch.tensor(out), 因为本身就是一个tensor, 这样反而会断开梯度
        return out
    
class RMSNorm(torch.nn.Module):
    """token level 的 Normalization"""
    def __init__(
            self, d_model: int, 
            eps: float=1e-5, device: 
            torch.device=None, 
            dtype: torch.dtype=None
            ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # RMSNorm weight should be initialized to 1
        self.weight = torch.nn.Parameter(torch.ones([d_model], device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        assert x.shape[-1] == self.d_model
        rms = self.eps
        rms += torch.sum((x ** 2 / self.d_model), dim=-1)
        rms = torch.sqrt(rms)
        rms = rearrange(rms, "b s -> b s 1")
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(
            self, d_model: int, 
            d_ff: int=0, 
            device: torch.device | None=None, 
            dtype: torch.dtype | None=None
            ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if d_ff == 0:
            target = (d_model * 8) // 3
            if target % 64 >= 32:
                d_ff = ((target // 64) + 1) * 64
            else:
                d_ff = (target // 64) * 64
            
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)
        x2 = self.w3(x)
        y = x1 * x2
        out = self.w2(y)
        return out

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)
        x2 = self.w3(x)
        y = x1 * x2
        return y


class RoPE(torch.nn.Module):
    def __init__(
            self, 
            theta: float, 
            d_k: int, 
            max_seq_len: int, 
            device: torch.device | None=None
            ):
        super().__init__()
        self.theta = theta
        # dim of query and key vectors
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device


        def construct_r():
            # 插入新的轴
            i = rearrange(torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32), "seq_len -> seq_len 1")
            # 好像不能搞成 1, self.d_k // 2 + 1
            k = rearrange(torch.arange(1, self.d_k, 2, device=self.device, dtype=torch.float32), "blk_cnt -> 1 blk_cnt")
            theta_i_k = i / (self.theta ** ((k - 1) / self.d_k))
            cos_vals = torch.cos(theta_i_k)
            sin_vals = torch.sin(theta_i_k)
            r = torch.stack([cos_vals, sin_vals], dim=-1)  # Shape: [max_seq_len, d_k // 2, 2]
            return r
        r_blk = construct_r()
        
        self.register_buffer("r_blk", r_blk, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        将r, x看成复数, 方便进行逐元素的矩阵乘法
        """
        dtype = x.dtype
        x = rearrange(x, "... seq_len (d two) -> ... seq_len d two", two=2).to(torch.float32).contiguous()
        x = torch.view_as_complex(x)
        r = self.r_blk[token_positions]
        r = rearrange(r, "... seq_len out_d in_d -> (... seq_len) out_d in_d")
        r = torch.view_as_complex(r)
        result = rearrange(torch.view_as_real(r * x), "... seq_len d two -> ... seq_len (d two)")
        return result.to(dtype)
    
def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    max_entry = torch.max(x, dim=i, keepdim=True).values
    x_ = x - max_entry
    x_ = torch.exp(x_)
    sum_entry = torch.sum(x_, dim=i, keepdim=True)
    out = x_ / sum_entry
    return out.to(dtype)

def scaled_dot_prod_atten(
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        mask: torch.Tensor | None=None
        ) -> torch.Tensor:
    """
    queries, keys: (batch_size, ..., q/k, d_k)
    values: (batch_size, ..., seq_len, d_v)
    mask: (q k)
    """
    # 记得乘缩放因子
    factor = queries.shape[-1] ** 0.5
    atten_weight = einsum(queries, keys, "... q d_k, ... k d_k -> ... q k") / factor
    if mask is not None:
        mask.to(dtype=torch.bool)
        hidden = torch.where(mask == True, 0.0, -torch.inf)
        atten_weight += hidden
    softmax_atten_weight = softmax(atten_weight, -1)
    atten = einsum(values, softmax_atten_weight, "... seq_len d_v, ... q seq_len -> ... q d_v")
    return atten

class Multihead_self_atten(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            device: torch.device | None=None, 
            dtype: torch.dtype | None=None
            ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.device = device
        self.dtype = dtype

        self.w_q = Linear(d_model, num_heads * self.d_k, dtype=dtype, device=device)
        self.w_k = Linear(d_model, num_heads * self.d_k, dtype=dtype, device=device)
        self.w_v = Linear(d_model, num_heads * self.d_v, dtype=dtype, device=device)
        self.w_o = Linear(num_heads * self.d_v, d_model, dtype=dtype, device=device)
        
    def forward(
            self, 
            x: torch.Tensor, 
            max_seq_len: int=0, 
            theta: float=0.0, 
            token_positions: torch.Tensor | None=None
            ) -> torch.Tensor:
        
        seq_len = x.shape[-2]
        device = x.device
        q = self.w_q(x)
        q_ = rearrange(q, "... seq_len (n d_k) -> ... n seq_len d_k", d_k=self.d_k)
        k = self.w_k(x)
        k_ = rearrange(k, "... seq_len (n d_k) -> ... n seq_len d_k", d_k=self.d_k)
        v = self.w_v(x)
        v_ = rearrange(v, "... seq_len (n d_v) -> ... n seq_len d_v", d_v=self.d_v)
        
        if max_seq_len != 0:
            # rope是旋转, 所以不是+=而是=
            rope = RoPE(theta, self.d_k, max_seq_len)
            q_ = rope(q_, token_positions)
            k_ = rope(k_, token_positions)
        # 位置i的query不能分配注意力给位置j的key(j>i)
        mask = torch.tril(torch.ones([seq_len, seq_len], device=device, dtype=torch.bool))
        h = rearrange(scaled_dot_prod_atten(q_, k_, v_, mask), "... n seq_len d_v -> ... seq_len (n d_v)")
        return self.w_o(h)

class Transformer_blk(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_ff: int,
            max_seq_len: int,
            theta: float,
            device: torch.device | None=None, 
            dtype: torch.dtype | None=None
            ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = Multihead_self_atten(d_model, num_heads, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        x_ = self.ln1(x)
        token_positions = torch.arange(seq_len)
        x_ = self.attn(x_, self.max_seq_len, self.theta, token_positions)
        y = x + x_

        y_ = self.ln2(y)
        y_ = self.ffn(y_)
        out = y + y_
        return out
        

class Transformer_LM(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            device: torch.device | None=None, 
            dtype: torch.dtype | None=None
            ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = torch.nn.ModuleList([Transformer_blk(d_model, num_heads, d_ff, context_length, theta, device, dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]) -> Float[torch.Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for transformer_blk in self.layers:
            x = transformer_blk(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

