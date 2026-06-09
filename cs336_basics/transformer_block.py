import torch
from torch import nn

from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        ''' x: (batch_size, seq_len, d_model)

            return: (batch_size, seq_len, d_model)
        '''
        x = x + self.mha(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x
