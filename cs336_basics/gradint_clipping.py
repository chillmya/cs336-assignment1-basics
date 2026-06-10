import torch
from typing import Iterable
import math


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_squared_norm: float = 0.0

    for p in parameters:
        if p.grad is None:
            continue
        total_squared_norm += torch.sum(p.grad ** 2)

    total_norm: float = math.sqrt(total_squared_norm)
    if total_norm > max_l2_norm:
        scale: float = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad *= scale