import math

def get_lr_cosine_schedule(it: int, warmup_iters: int, cosine_cycle_iters: int, max_lr: float, min_lr: float):

    if (it < warmup_iters):
        return max_lr * it / warmup_iters
    
    elif (it > cosine_cycle_iters):
        return min_lr
    
    else:
        # Cosine decay phase
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
