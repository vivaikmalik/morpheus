"""
Proximal Policy Optimisation (PPO) — clipped surrogate variant.

Runs multiple optimisation epochs on the same batch of generated molecules.
The training script must call compute_loss() multiple times per batch
(controlled by `ppo_epochs`).

loss = -mean(min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)) + β * KL

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
"""

import torch
from rl.algorithms.base import RLAlgorithm, RLStepOutput


class PPO(RLAlgorithm):

    def __init__(
        self,
        beta: float = 0.1,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        max_adv: float = 3.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        beta       : KL penalty weight
        clip_eps   : clipping range for the importance ratio
        ppo_epochs : number of optimisation passes per batch of generated data
        max_adv    : clamp magnitude for advantages
        """
        super().__init__(beta=beta, **kwargs)
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.max_adv = max_adv

        # Set by pre_step / first compute_loss call
        self._old_log_probs = None
        self._advantages = None
        self._ref_log_probs = None
        self._epoch_idx = 0

    @property
    def needs_multiple_generations(self) -> bool:
        return True

    def pre_step(self, step: int, **kwargs):
        self._old_log_probs = None
        self._advantages = None
        self._ref_log_probs = None
        self._epoch_idx = 0

    def compute_loss(
        self,
        policy_log_probs: torch.Tensor,   # (B,) with grad
        rewards: torch.Tensor,             # (B,) — only used on first epoch
        ref_log_probs: torch.Tensor,       # (B,) — only used on first epoch
        **kwargs,
    ) -> RLStepOutput:

        # On first epoch: freeze the old log-probs and compute advantages
        if self._old_log_probs is None:
            self._old_log_probs = policy_log_probs.detach().clone()
            self._advantages = (rewards - rewards.mean()).clamp(
                -self.max_adv, self.max_adv
            )
            self._ref_log_probs = ref_log_probs.detach().clone()

        advantage = self._advantages

        # ── Clipped surrogate ────────────────────────────────────────
        ratio = torch.exp(policy_log_probs - self._old_log_probs)
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

        # ── KL penalty ───────────────────────────────────────────────
        kl_penalty = (policy_log_probs - self._ref_log_probs).clamp(min=0)

        loss = -surrogate.mean() + self.beta * kl_penalty.mean()

        self._epoch_idx += 1

        return RLStepOutput(
            loss=loss,
            metrics={
                "mean_kl": kl_penalty.mean().item(),
                "mean_advantage": advantage.mean().item(),
                "mean_ratio": ratio.mean().item(),
                "clip_fraction": ((ratio - 1.0).abs() > self.clip_eps).float().mean().item(),
                "ppo_epoch": self._epoch_idx,
            },
        )

    def post_step(self, step: int, **kwargs):
        self._old_log_probs = None
        self._advantages = None
        self._ref_log_probs = None
        self._epoch_idx = 0
