
import torch
import numpy as np
from einops import rearrange, einsum


images = torch.tensor([1, 2])  

dim_value = rearrange(images,    "b   ->  1 b")

print(dim_value) 

