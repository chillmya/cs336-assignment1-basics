
import torch
import numpy as np
from einops import rearrange, einsum


x  = torch.tensor(torch.arange(12))
print(x)

x = rearrange(x, '... (n two) -> ... n two', two=2)
print(x)

x1 = x[..., 0]
print(x1)
