import torch
import torch.nn as nn
from einops import rearrange

from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import Attention
from cs336_basics.rope import RoPE


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
        rope_theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if rope_theta is not None:
            assert max_seq_len is not None
            self.rope = RoPE(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        x: (..., seq_len, d_model)
        token_positions: (..., seq_len) | None

        return: (..., seq_len, d_model)
        """

        # q, k, v: (..., num_heads, seq_len, d_k) where d_k = d_model / num_heads
        q = rearrange(self.W_q(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        k = rearrange(self.W_k(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        v = rearrange(self.W_v(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)

        seq_len = x.shape[-2]

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            else:
                token_positions = token_positions.to(device=x.device)
            rope_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        # attn_output: (..., num_heads, seq_len, d_v)
        attn_output = Attention(q, k, v, causal_mask)

        # multi_head_output: (..., seq_len, d_model)
        multi_head_output = rearrange(attn_output, "... num_heads seq d_v -> ... seq (num_heads d_v)")

        # output: (..., seq_len, d_model)
        output = self.W_o(multi_head_output)

        return output
