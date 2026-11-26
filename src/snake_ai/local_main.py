"""Runs Snake in a local Tk window, optionally training a new policy first.

Models are loaded from and saved to the model directory from the settings
(``SNAKE_AI_MODEL_PATH``, by default ``policy_models/`` in the repository
root), so this works from any working directory::

    uv run python -m snake_ai.local_main                  # play with dqn-1m.pt
    uv run python -m snake_ai.local_main --model FILE     # play with FILE
    uv run python -m snake_ai.local_main --train          # train, then play
    uv run python -m snake_ai.local_main --seed N         # use seed N (default 42)
"""

import argparse
import copy

import torch

from snake_ai.ai import DQNTrainer, Policy
from snake_ai.gui import Controller, View
from snake_ai.session import Game
from snake_ai.utils import configure_logging, get_settings


def main(train: bool, model_file: str, seed: int, **kwargs):
    """Sets up logging, prepares a policy and opens the Tk GUI.

    With ``train``, a fresh policy is trained with ``DQNTrainer`` on a
    headless ``Game``. Checkpoints after each evaluation and the final
    model are saved to timestamped files in the model directory
    (``Settings.model_path``). Otherwise, the weights are loaded
    from ``model_file``. Either way, the GUI then plays with the resulting
    policy.

    Args:
        train: Whether to train a new policy instead of loading one.
        model_file: The model file, relative to the model directory. An
            absolute path is used as is. Ignored when ``train`` is set.
        seed: Seeds torch, the game's food placement and, with ``train``, the
            trainer's exploration, so that runs are reproducible.
        **kwargs: Hyperparameters passed on to ``DQNTrainer`` (``lr``,
            ``warm_up``, ``train_steps``, ``gamma``, ``batch_size``,
            ``update_target``, ``update_policy``). Ignored when ``train``
            is not set.

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
        agent = DQNTrainer(policy, seed=seed, **kwargs)
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
        default="dqn-1m.pt",
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

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help=(
            "Learning rate parameter of the optimizer (default: %(default)s)"
        )
    )

    parser.add_argument(
        "--warm-up",
        type=int,
        default=10_000,
        help=(
            "Number of steps before training starts (default: %(default)s)"
        )
    )

    parser.add_argument(
        "--train-steps",
        type=int,
        default=3_000_000,
        help=(
            "Number of training steps (default: %(default)s)"
        )
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "Future reward penalty (default: %(default)s)"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Training batch size (default: %(default)s)"
        )
    )

    parser.add_argument(
        "--update-target",
        type=int,
        default=1_000,
        help=(
            "Number of steps between target network updates (default: %(default)s)"
        )
    )

    parser.add_argument(
        "--update-policy",
        type=int,
        default=4,
        help=(
            "Number of steps between policy network updates (default: %(default)s)"
        )
    )

    args = parser.parse_args()

    training_args = copy.deepcopy(vars(args))
    del training_args["train"]
    del training_args["model"]
    del training_args["seed"]
    main(train=args.train, model_file=args.model, seed=args.seed, **training_args)
