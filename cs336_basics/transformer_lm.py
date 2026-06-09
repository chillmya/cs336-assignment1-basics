import torch
from torch import nn

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding

class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        context_len: int,
        vocab_size: int,
        num_layers: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                max_seq_len=context_len,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ''' token_ids: (batch_size, seq_len)

            return: (batch_size, seq_len, vocab_size)
        '''
        x = self.token_embedding(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.output_embedding(x)

        return logits
    
