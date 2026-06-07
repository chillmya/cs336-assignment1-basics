import torch
import torch.nn as nn
from einops import rearrange


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        assert d_k % 2 == 0

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, device=device)  # (max_seq_len,)
        dim_indices = torch.arange(0, d_k, 2, device=device)   # (d_k / 2,)

        inv_freq = 1.0 / (theta ** (dim_indices.float() / d_k))

        # angles: (max_seq_len, d_k / 2)
        angles = positions[:, None].float() * inv_freq[None, :]

        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)

        return: (..., seq_len, d_k)
        """

        # x_pair: (..., seq_len, d_k / 2, 2)
        x_pair = rearrange(x, "... seq_len (d_pair two) -> ... seq_len d_pair two", two=2)

        x1 = x_pair[..., 0]  # (..., seq_len, d_k / 2)
        x2 = x_pair[..., 1]  # (..., seq_len, d_k / 2)

        cos = self.cos[token_positions]  # (..., seq_len, d_k / 2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k / 2)

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # (..., seq_len, d_k / 2, 2)
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)

        # (..., seq_len, d_k)
        return rearrange(rotated, "... seq_len d_pair two -> ... seq_len (d_pair two)")