"""Runs Snake in a local Tk window, optionally training a new policy first.

Models are loaded from and saved to the model directory from the settings
(``SNAKE_AI_MODEL_PATH``, by default ``policy_models/`` in the repository
root), so this works from any working directory::

    uv run python -m snake_ai.local_main                  # play with model.pt
    uv run python -m snake_ai.local_main --model FILE     # play with FILE
    uv run python -m snake_ai.local_main --train          # train, then play
    uv run python -m snake_ai.local_main --seed N         # use seed N (default 42)
"""

import argparse

import torch

from snake_ai.ai import DQNTrainer, Policy
from snake_ai.gui import Controller, View
from snake_ai.snake import Game
from snake_ai.utils import configure_logging, get_settings


def main(train: bool, model_file: str, seed: int):
    """Sets up logging, prepares a policy and opens the Tk GUI.

    With ``train``, a fresh policy is trained with ``DQNTrainer`` on a
    headless ``Game`` and saved to a timestamped file in the model
    directory (``Settings.model_path``). Otherwise, the weights are loaded
    from ``model_file``. Either way, the GUI then plays with the resulting
    policy.

    Args:
        train: Whether to train a new policy instead of loading one.
        model_file: The model file, relative to the model directory. An
            absolute path is used as is. Ignored when ``train`` is set.
        seed: Seeds torch, the game's food placement and, with ``train``, the
            trainer's exploration, so that runs are reproducible.

    Raises:
        FileNotFoundError: If ``train`` is not set and the model file does
            not exist.
    """
    torch.manual_seed(seed)

    settings = get_settings()
    configure_logging()

    if train:
        policy = Policy()
    else:
        policy = Policy.load_model(
            model_file=model_file, model_path=settings.model_path)

    game = Game(policy)
    game.reset(seed=seed)

    if train:
        agent = DQNTrainer(policy, seed=seed)
        agent.train(game, model_path=settings.model_path)

    Controller(game, View()).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Snake AI",
        description="AI agent that learns to play Snake through Reinforcement Learning"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="model.pt",
        metavar="FILE",
        help=(
            "Model file in the model directory, or an absolute path "
            "(default: %(default)s)"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random number generator seed that is used to ensure reproducible runs "
            "(default: %(default)s)"
        )
    )

    args = parser.parse_args()

    main(train=args.train, model_file=args.model, seed=args.seed)
