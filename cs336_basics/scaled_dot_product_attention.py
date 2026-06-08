import torch
import torch.nn as nn
from einops import rearrange, einsum

def Attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    q, k, v: (batch_size, seq_len, d_k)

    return: (batch_size, seq_len, d_k)
    """

    d_k = q.shape[-1]

    # attn_scores: (batch_size, seq_len, seq_len)
    attn_scores = einsum(q, k, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / (d_k ** 0.5)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(attn_scores, dim=-1)

    # output: (batch_size, seq_len, d_k)
    output = einsum(attn_weights, v, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

    return output


