import torch
from einops import rearrange, reduce


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (..., vocab)
    targets: (...)
    
    return: scalar loss
    """

    # targets 必须是整数 token id
    targets = targets.long()

    # 1. 对 vocab 维度取 max
    # logits:     (..., vocab)
    # max_logits: (...)
    max_logits = reduce(logits, "... vocab -> ...", "max")

    # 2. 把 max_logits 变成 (..., 1)，方便和 (..., vocab) 相减
    # max_logits_for_broadcast: (..., 1)
    max_logits_for_broadcast = rearrange(max_logits, "... -> ... 1")

    # 3. 数值稳定：所有 logits 减去同一个 max，不改变 softmax 结果
    # shifted_logits: (..., vocab)
    shifted_logits = logits - max_logits_for_broadcast

    # 4. 计算 log(sum(exp(shifted_logits)))
    # exp_shifted_logits: (..., vocab)
    # sum_exp:            (...)
    # log_sum_exp:        (...)
    exp_shifted_logits = torch.exp(shifted_logits)
    sum_exp = reduce(exp_shifted_logits, "... vocab -> ...", "sum")
    log_sum_exp = torch.log(sum_exp)

    # 5. 从 vocab 维度中取出正确 target 对应的 logit
    # targets:        (...)
    # target_indices: (..., 1)
    target_indices = rearrange(targets, "... -> ... 1")

    # gathered: (..., 1)
    gathered = torch.gather(
        shifted_logits,
        dim=-1,
        index=target_indices,
    )

    # target_logits: (...)
    target_logits = rearrange(gathered, "... 1 -> ...")

    # 6. CE = logsumexp - correct_logit
    # losses: (...)
    losses = log_sum_exp - target_logits

    # 7. 对所有 batch-like dimensions 求平均
    return losses.mean()