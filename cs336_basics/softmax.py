import torch
import torch.nn as nn
from einops import rearrange

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_k)
        # return: (..., d_k)

        # max_x: (...,)
        max_x = torch.max(x, dim=self.dim, keepdim=True).values

        # exp_x: (..., d_k)
        exp_x = torch.exp(x - max_x)

        # sum_exp_x: (...,)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)

        return exp_x / sum_exp_x