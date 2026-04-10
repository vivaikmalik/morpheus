"""
Group Relative Policy Optimisation (GRPO).

Instead of a single sample per prompt, GRPO generates G samples per prompt
and uses within-group ranking to compute advantages. This eliminates the need
for a learned value function while reducing variance.

Reference: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models", 2024.

loss = -mean(advantage * min(ratio, clip(ratio, 1-ε, 1+ε))) + β * KL

where:
    ratio     = exp(log π_θ - log π_old)        (importance weight)
    advantage = (reward - group_mean) / group_std   (normalised within group)
    KL        = max(0, log π_θ - log π_ref)
"""

import torch
from rl.algorithms.base import RLAlgorithm, RLStepOutput


class GRPO(RLAlgorithm):

    def __init__(
        self,
        beta: float = 0.1,
        group_size: int = 4,
        clip_eps: float = 0.2,
        **kwargs,
    ):
        """
        Parameters
        ----------
        beta       : KL penalty weight
        group_size : G — number of samples generated per prompt.
                     The training script is responsible for generating G samples
                     per prompt and passing them in grouped order.
        clip_eps   : PPO-style clipping range for the importance ratio
        """
        super().__init__(beta=beta, **kwargs)
        self.group_size = group_size
        self.clip_eps = clip_eps
        self._old_log_probs = None

    @property
    def needs_multiple_generations(self) -> bool:
        return False  # generates G at once, not multiple passes

    def pre_step(self, step: int, **kwargs):
        self._old_log_probs = None

    def compute_loss(
        self,
        policy_log_probs: torch.Tensor,   # (B,) where B = num_prompts * group_size
        rewards: torch.Tensor,             # (B,)
        ref_log_probs: torch.Tensor,       # (B,)
        **kwargs,
    ) -> RLStepOutput:
        B = policy_log_probs.shape[0]
        G = self.group_size

        if B % G != 0:
            raise ValueError(
                f"Batch size {B} must be divisible by group_size {G}. "
                f"Pass --batch_size as a multiple of --group_size."
            )

        # Store old log-probs on first call (before gradient step)
        if self._old_log_probs is None:
            self._old_log_probs = policy_log_probs.detach().clone()

        # ── Group-relative advantage ─────────────────────────────────
        # Reshape to (num_groups, G) for within-group normalisation
        rewards_grouped = rewards.view(-1, G)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        # Avoid dividing by group_std. With small G=4, group_std can be tiny by chance
        # and cause massive, destructive gradient explosions. 
        # DeepSeekMath GRPO often omits std division for safety, or uses it only with huge G.
        advantage = (rewards_grouped - group_mean).view(B)

        # ── Clipped importance ratio ─────────────────────────────────
        ratio = torch.exp(policy_log_probs - self._old_log_probs)
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

        # ── KL penalty ───────────────────────────────────────────────
        kl_penalty = (policy_log_probs - ref_log_probs).clamp(min=0)

        loss = -surrogate.mean() + self.beta * kl_penalty.mean()

        return RLStepOutput(
            loss=loss,
            metrics={
                "mean_kl": kl_penalty.mean().item(),
                "mean_advantage": advantage.mean().item(),
                "mean_ratio": ratio.mean().item(),
                "clip_fraction": ((ratio - 1.0).abs() > self.clip_eps).float().mean().item(),
            },
        )

    def post_step(self, step: int, **kwargs):
        self._old_log_probs = None
