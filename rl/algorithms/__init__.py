"""
Algorithm registry.

Usage:
    from rl.algorithms import get_algorithm
    algo = get_algorithm("reinforce", beta=0.1)
"""

from rl.algorithms.base import RLAlgorithm, RLStepOutput
from rl.algorithms.reinforce import REINFORCE
from rl.algorithms.grpo import GRPO
from rl.algorithms.ppo import PPO

ALGORITHM_REGISTRY: dict[str, type[RLAlgorithm]] = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
    "ppo": PPO,
}


def get_algorithm(name: str, **kwargs) -> RLAlgorithm:
    """Instantiate an algorithm by name. Extra kwargs are forwarded to __init__."""
    key = name.lower()
    if key not in ALGORITHM_REGISTRY:
        available = ", ".join(sorted(ALGORITHM_REGISTRY.keys()))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return ALGORITHM_REGISTRY[key](**kwargs)


def list_algorithms() -> list[str]:
    return sorted(ALGORITHM_REGISTRY.keys())
