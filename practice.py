
import torch
import numpy as np
from einops import rearrange, einsum, reduce


x  = torch.tensor(torch.arange(12))
print(x)

x = rearrange(x, "... -> ... 1")
x = rearrange(x, "... 1 -> ...")

max_logits = reduce(x, "... vocab -> ...", "max")
print(max_logits)
