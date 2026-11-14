"""Runs Snake in a local Tk window, optionally training a new policy first.

Run it from the repository root, since models are looked up in
``policy_models/`` relative to the current working directory::

    uv run python -m snake_ai.local_main                  # play with model.pt
    uv run python -m snake_ai.local_main --model FILE     # play with FILE
    uv run python -m snake_ai.local_main --train          # train, then play
"""

import argparse

from snake_ai.ai import DQNTrainer, Policy
from snake_ai.gui import Controller, View
from snake_ai.snake import Game
from snake_ai.utils import configure_logging


def main(train: bool, model_file: str):
    """Sets up logging, prepares a policy and opens the Tk GUI.

    With ``train``, a fresh policy is trained with ``DQNTrainer`` on a
    headless ``Game`` and saved to a timestamped file in
    ``MODEL_BASE_PATH``. Otherwise, the weights are loaded from
    ``model_file``. Either way, the GUI then plays with the resulting policy.

    Args:
        train: Whether to train a new policy instead of loading one.
        model_file: The file name of the model inside ``MODEL_BASE_PATH``.
            Ignored when ``train`` is set.

    Raises:
        FileNotFoundError: If ``train`` is not set and the model file does
            not exist.
    """
    configure_logging()
    
    if train:
        policy = Policy()
    else:
        policy = Policy.load_model(model_file=model_file)
    
    game = Game(policy)

    if train:
        agent = DQNTrainer(policy)
        agent.train(game)

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
        help="File name of the model in policy_models/ (default: %(default)s)"
    )

    args = parser.parse_args()

    main(train=args.train, model_file=args.model)
