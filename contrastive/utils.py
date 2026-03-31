import torch

def mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    masked = last_hidden * mask
    summed = masked.sum(dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom
