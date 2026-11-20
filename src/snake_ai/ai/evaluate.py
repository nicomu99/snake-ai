import numpy as np
from tqdm import tqdm

from snake_ai.ai import Policy
from snake_ai.snake import Game


def evaluate(
    policy: Policy,
    n_episodes: int = 50,
    max_steps: int = 500,
    seed: int = 10
) -> dict:
    """Evaluation script to quantify the performance of a policy.

    Computes the following metrics:
        - food_mean:
        - win_rate: relative number of times the policy won games
        - collision_rate: relative number of times the policy bit itself
        - reversal_rate: relative number of times the policy tried reversing
            the direction
        - timeout_rate: ?
    """
    if n_episodes < 1:
        raise ValueError(
            f"Invalid number of episodes {n_episodes}, must be larger than 0")

    policy = policy.eval()
    game = Game(policy)
    game.reset(seed)
    seed += 1

    score_list = []
    win, collision, reversal, timeout, total_steps = 0, 0, 0, 0, 0
    actions = []
    for _ in tqdm(range(n_episodes)):
        while not game.game_finished and game.step_count < max_steps:
            action = policy.act(game.snake_env.get_state())
            game.step(action)
            actions.append(action)

        score = game.snake_env.score
        score_list.append(score)
        total_steps += game.step_count

        env_state = game.snake_env.get_game_state()
        if env_state == "game_won":
            win += 1
        elif env_state == "collision":
            collision += 1
        elif env_state == "direction_reversal":
            reversal += 1
        else:
            timeout += 1
        game.reset(seed=seed)
        seed += 1

    score_list = np.array(score_list)
    food_sum = np.sum(score_list)
    steps_per_food = total_steps / food_sum if food_sum > 0 else 0
    metrics_dict = {
        "eval/food_mean": np.mean(score_list),
        "eval/food_median": np.median(score_list),
        "eval/food_max": np.max(score_list),
        "eval/win_rate": win / n_episodes,
        "eval/collision_rate": collision / n_episodes,
        "eval/reversal_rate": reversal / n_episodes,
        "eval/timeout_rate": timeout / n_episodes,
        "eval/steps_per_food": steps_per_food,
        "eval/action_hist": actions,
    }
    return metrics_dict
