"""
REINFORCE with KL-regularised baseline subtraction.

loss = -mean(advantage * log_prob) + β * mean(KL_penalty)

where:
    advantage  = reward - mean(reward),  clamped to [-max_adv, max_adv]
    KL_penalty = max(0, log π_θ - log π_ref)
"""

import torch
from rl.algorithms.base import RLAlgorithm, RLStepOutput


class REINFORCE(RLAlgorithm):

    def __init__(self, beta: float = 0.1, max_adv: float = 3.0, **kwargs):
        super().__init__(beta=beta, **kwargs)
        self.max_adv = max_adv

    def compute_loss(
        self,
        policy_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        ref_log_probs: torch.Tensor,
        **kwargs,
    ) -> RLStepOutput:
        advantage = (rewards - rewards.mean()).clamp(-self.max_adv, self.max_adv)
        kl_penalty = (policy_log_probs - ref_log_probs).clamp(min=0)

        loss = -(advantage * policy_log_probs).mean() + self.beta * kl_penalty.mean()

        return RLStepOutput(
            loss=loss,
            metrics={
                "mean_kl": kl_penalty.mean().item(),
                "mean_advantage": advantage.mean().item(),
                "std_advantage": advantage.std().item(),
            },
        )
