"""
Abstract base class for RL algorithms.

Every algorithm must implement `compute_loss()` and can optionally override
`pre_step()` / `post_step()` hooks for algorithms that need state between
steps (e.g. PPO's old log-probs).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class RLStepOutput:
    """Standard return type from compute_loss()."""
    loss: torch.Tensor
    metrics: dict  # algorithm-specific scalars for logging


class RLAlgorithm(ABC):
    """
    Interface that every RL algorithm must satisfy.

    The training loop calls:
        1. algo.pre_step(...)      — optional setup before generation
        2. [generation happens]
        3. algo.compute_loss(...)   — returns loss + metrics
        4. [optimizer.step()]
        5. algo.post_step(...)     — optional bookkeeping after update
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        """
        Parameters
        ----------
        beta : float
            KL penalty weight against the reference model.
        """
        self.beta = beta

    @abstractmethod
    def compute_loss(
        self,
        policy_log_probs: torch.Tensor,   # (B,) with grad
        rewards: torch.Tensor,             # (B,) no grad
        ref_log_probs: torch.Tensor,       # (B,) no grad
        **kwargs,
    ) -> RLStepOutput:
        """
        Compute the RL loss given log-probs and rewards.

        Parameters
        ----------
        policy_log_probs : (B,) accumulated log π_θ for each sample, with gradients
        rewards          : (B,) scalar reward per sample
        ref_log_probs    : (B,) accumulated log π_ref for each sample, no gradients

        Returns
        -------
        RLStepOutput with .loss (scalar Tensor) and .metrics (dict)
        """
        ...

    def pre_step(self, step: int, **kwargs):
        """Called before generation. Override for algorithms that need setup."""
        pass

    def post_step(self, step: int, **kwargs):
        """Called after optimizer.step(). Override for bookkeeping."""
        pass

    @property
    def needs_multiple_generations(self) -> bool:
        """Whether the algorithm needs multiple generations per step (e.g. PPO epochs)."""
        return False
