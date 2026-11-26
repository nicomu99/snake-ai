"""Greedy evaluation of a ``Policy`` on a fixed set of seeded games."""

import numpy as np
from tqdm import tqdm

from snake_ai.ai.policy import Policy
from snake_ai.snake import SnakeEnv

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def evaluate(
    policy: Policy,
    n_episodes: int = 50,
    max_steps: int = 500,
    seed: int = 10,
    progress: bool = False,
) -> dict:
    """Plays ``n_episodes`` greedy games and measures how well the policy does.

    Episode ``i`` is seeded with ``seed + i``, so repeated calls play the same
    boards and their results are comparable over the course of training. The
    policy is put into eval mode while playing, and its previous mode is
    restored afterward.

    Args:
        policy: The policy to evaluate.
        n_episodes: The number of games to play. Must be at least 1.
            Defaults to 50.
        max_steps: The step cap per game. A game that reaches it without
            ending counts as a timeout. Defaults to 500.
        seed: The seed of the first game. Defaults to 10.
        progress: Whether to show a progress bar. Defaults to False. When
            another bar is open (e.g. the trainer's), it is drawn below it
            and cleared once the evaluation finishes.

    Returns:
        A dict of metrics, keyed with an ``eval/`` prefix:

        - ``food_mean``, ``food_median``, ``food_max``: The amount of food
          eaten per game.
        - ``win_rate``: The share of games that were won.
        - ``collision_rate``: The share of games lost by the snake running
          into itself.
        - ``reversal_rate``: The share of games lost by reversing direction.
        - ``timeout_rate``: The share of games that reached ``max_steps``
          without ending.
        - ``steps_per_food``: The total number of steps divided by the total
          amount of food eaten, or 0 if no food was eaten. Steps after the
          last food of a game are included, so timeouts make it larger.

    Raises:
        ValueError: If ``n_episodes`` is less than 1.
    """
    if n_episodes < 1:
        raise ValueError(
            f"Invalid number of episodes {n_episodes}, must be larger than 0")

    was_training = policy.training
    policy.eval()
    snake_env = SnakeEnv()

    score_list = []
    win, collision, reversal, timeout, total_steps = 0, 0, 0, 0, 0
    try:
        # leave=None keeps the bar only when it's the outermost one, so a
        # nested eval bar under the trainer's bar is cleared when done.
        for episode in tqdm(range(n_episodes), desc="eval", leave=None,
                            disable=not progress):
            snake_env.reset(seed=seed + episode)
            while (
                    not snake_env.lost and
                    not snake_env.won and
                    snake_env.step_count < max_steps
            ):
                action = policy.act(snake_env.get_state())
                action = ACTIONS[action]
                snake_env.perform_move(action)

            score_list.append(snake_env.score)
            total_steps += snake_env.step_count

            env_state = snake_env.get_game_state()
            if env_state == "game_won":
                win += 1
            elif env_state == "collision":
                collision += 1
            elif env_state == "direction_reversal":
                reversal += 1
            else:
                timeout += 1
    finally:
        policy.train(was_training)

    score_list = np.array(score_list)
    food_sum = np.sum(score_list)
    steps_per_food = total_steps / food_sum if food_sum > 0 else 0
    return {
        "eval/food_mean": np.mean(score_list),
        "eval/food_median": np.median(score_list),
        "eval/food_max": np.max(score_list),
        "eval/win_rate": win / n_episodes,
        "eval/collision_rate": collision / n_episodes,
        "eval/reversal_rate": reversal / n_episodes,
        "eval/timeout_rate": timeout / n_episodes,
        "eval/steps_per_food": steps_per_food,
    }
