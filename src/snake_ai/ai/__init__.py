"""The Snake-playing policy and its Double DQN trainer."""

from .dqn_trainer import DQNTrainer
from .policy import Policy

__all__ = [
    "DQNTrainer",
    "Policy",
]
