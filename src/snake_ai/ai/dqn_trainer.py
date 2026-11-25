"""Double DQN training for a ``Policy``."""

import copy
import logging
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from random import Random

import torch
from torch.nn import SmoothL1Loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from snake_ai.ai.evaluate import evaluate
from snake_ai.ai.policy import Policy

logger = logging.getLogger(__name__)


class DQNTrainer:
    """Trains a ``Policy`` with Double DQN, experience replay and epsilon-greedy
    exploration.

    The trainer wraps the given policy and keeps a frozen copy of it as the
    target network. The policy is trained and picks actions. The target
    network is synced every ``update_target_steps`` steps and is used to
    compute stable training targets. Following Double DQN, the policy
    chooses the next action and the target network scores it.

    The trainer is only needed for training. For inference, use
    ``Policy.load_model`` and ``Policy.act`` directly.

    Attributes:
        rng: A random number generator for choosing random actions during training.
        policy: The policy that is trained and used to pick actions. It is
            updated in place.
        target_model: A frozen, periodically synced copy of ``policy`` used
            for training targets.
        optimizer: Adam optimizer for the policy network.
        criterion: Smooth L1 (Huber) loss between predicted and target
            Q-values. It is less sensitive to large TD errors than MSE.
        experience_replay: A bounded buffer of
            ``(state, action, reward, next_state, done)`` transitions.
        batch_size: The number of transitions per training update.
        epsilon: The current probability of taking a random action. It
            decays linearly from ``epsilon_start`` to ``epsilon_min`` over
            ``epsilon_decay_steps`` steps and then stays at ``epsilon_min``.
        epsilon_start: The initial value of ``epsilon``.
        epsilon_min: The lower bound for ``epsilon``.
        epsilon_decay_steps: The number of steps over which ``epsilon``
            decays to ``epsilon_min``.
        gamma: The discount factor for future rewards.
        train_steps: The total number of environment steps in ``train``.
        warm_up_steps: Steps to collect before the gated training updates
            begin.
        update_target_steps: How often (in steps) the target network is synced.
        train_policy_steps: How often (in steps) the gated training update
            runs once ``warm_up_steps`` is reached.
        eval_steps: How often (in steps) the policy is evaluated with
            ``evaluate``.
        eval_episodes: The number of games per evaluation.
        eval_max_steps: The step cap per evaluation game.
        steps: The number of environment steps taken so far.
        device: The device of ``policy``.
        log_train_updates: The number of training updates whose metrics are
            averaged before they are written.
        summary_writer: The TensorBoard writer that metrics are written to.
            It writes to ``runs/<date>_<host>_lr<lr>_steps_<train_steps>``
            relative to the current working directory.
    """

    def __init__(
        self,
        policy: Policy,
        seed: int = 11,
        lr: float = 0.001,
        warm_up: int = 10_000,
        train_steps: int = 1_000_000,
        gamma: float = 0.95,
        batch_size: int = 256,
        update_target: int = 1_000,
        update_policy: int = 4,
    ):
        """Sets up the target network, optimizer and training hyperparameters.

        Args:
            policy: The policy to train. It can be freshly initialized or
                already loaded with weights to continue training. The target
                network starts as a copy of it.
            seed: Random number generator seed. Defaults to 11.
            lr: The learning rate of the Adam optimizer. Defaults to 0.001.
            warm_up: The number of steps to collect into the replay buffer
                before training updates begin. Defaults to 10_000.
            train_steps: The total number of environment steps to train
                for. Defaults to 1_000_000.
            gamma: The discount factor for future rewards. Defaults to 0.95.
            batch_size: The number of transitions per training update.
                Defaults to 256.
            update_target: How often (in steps) the target network is
                synced with the policy. Defaults to 1_000.
            update_policy: How often (in steps) a training update runs once
                the warm-up is over. Defaults to 4.
        """
        logger.info("Initializing agent")

        self.rng = Random(seed)

        self.policy = policy
        self.target_model = copy.deepcopy(self.policy)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        self.optimizer = Adam(
            self.policy.parameters(), lr=lr)
        self.criterion = SmoothL1Loss()
        self.experience_replay = deque(maxlen=500_000)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon_start = 1.0
        self.epsilon_min = 0.1
        self.epsilon = 1.0
        self.epsilon_decay_steps = 500_000

        self.train_steps = train_steps
        self.warm_up_steps = warm_up

        self.eval_steps = 10_000
        self.eval_episodes = 50
        self.eval_max_steps = 3000

        # Variables for updating the target model periodically
        self.update_target_steps = update_target
        self.train_policy_steps = update_policy
        self.steps = 0

        self.device = self.policy.device
        # Train metrics are averaged over this many updates before writing
        self.log_train_updates = 100

        self.summary_writer = SummaryWriter(comment=f"_lr{lr}_steps_{train_steps}")

    def train(self, env, model_path: str | Path):
        """Trains the policy and saves its weights.

        Runs ``run_train`` and then saves the policy's state dict to
        ``model_path / "<timestamp>_model_<train_steps>.pt"``, with the
        timestamp formatted as ``%Y-%m-%d_%H:%M:%S``. The directory is
        created if needed. The environment is reset afterward.

        Args:
            env: The environment to train on, usually a ``Game``. See
                ``run_train`` for what it must provide.
            model_path: The directory the trained model is saved to.
        """
        # Train the agent if no model is found
        logger.info("Starting agent training")
        self.run_train(env)

        logger.info("Finished training agent")

        # Save the trained model
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_file = model_path / f"{timestamp}_model_{self.train_steps}.pt"
        torch.save(self.policy.state_dict(), model_file)
        env.reset()

    def update_target_model(self):
        """Copies the policy network's weights into the target network."""
        self.target_model.load_state_dict(self.policy.state_dict())

    def play_one_step(self, state, env):
        """Picks an epsilon-greedy action and applies it to the environment.

        With probability ``epsilon`` a random action is taken, otherwise the
        policy's greedy action. Neither can reverse the snake's direction:
        random actions are drawn again while ``env.is_invalid_move`` is True
        (at most one of the four actions is a reversal, so this ends), and
        ``Policy.get_move`` masks the reversal. Reversals therefore never
        end a training episode.

        Args:
            state: The current encoded state.
            env: The environment to step. It must provide
                ``is_invalid_move(action)`` and ``step(action)``.

        Returns:
            A ``(reward, new_state, action, done)`` tuple, where ``new_state``
            is already encoded.
        """
        if self.rng.uniform(0, 1) < self.epsilon:
            action = self.rng.randint(0, 3)
            while env.is_invalid_move(action):
                action = self.rng.randint(0, 3)
        else:
            action = self.policy.get_move(state)

        reward, new_state, lost, won = env.step(action)
        new_state = self.policy.encode_state(new_state)

        return reward, new_state, action, lost or won

    def run_train(self, env):
        """Runs the training loop for ``train_steps`` environment steps.

        Each step plays one epsilon-greedy move, stores the transition,
        decays ``epsilon``, and may run a training update, a target network
        sync and an evaluation. Episode metrics are written when an episode
        ends, and the episode score average is logged every
        ``10 * update_target_steps`` steps. The summary writer is closed at
        the end.

        Args:
            env: The environment to train on, usually a ``Game``. It must
                provide ``reset()``, ``step(action)``,
                ``is_invalid_move(action)``, ``step_count`` and
                ``snake_env`` (for the episode metrics).
        """
        state = self.policy.encode_state(env.reset())
        score = []
        episode = 0
        episode_score = 0
        recent_outcomes = deque(maxlen=100)
        train_metrics = defaultdict(list)
        with logging_redirect_tqdm():
            for _ in tqdm(range(self.train_steps)):
                reward, next_state, action, done = self.play_one_step(state, env)

                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                self.steps += 1
                episode_score += reward

                # decay epsilon
                frac = min(self.steps / self.epsilon_decay_steps, 1.0)
                self.epsilon = self.epsilon_start - frac * (
                        self.epsilon_start - self.epsilon_min)

                if done:
                    recent_outcomes.append(env.snake_env.get_game_state())
                    counts = Counter(recent_outcomes)
                    n = len(recent_outcomes)

                    episode += 1
                    episode_dict = {
                        "episode/return": episode_score,
                        "episode/food": env.snake_env.score,
                        "episode/length": env.step_count,
                        "episode/win_rate": counts["game_won"] / n,
                        "episode/collision_rate": counts["collision"] / n,
                        "episode/reversal_rate": counts["direction_reversal"] / n,
                        "replay/size": len(self.experience_replay),
                        "explore/epsilon": self.epsilon,
                    }
                    self.write_metrics(episode_dict)

                    state = self.policy.encode_state(env.reset())
                    score.append(episode_score)
                    episode_score = 0

                if (
                    self.steps > self.warm_up_steps and
                    self.steps % self.train_policy_steps == 0
                ):
                    for k, v in self.train_from_experience().items():
                        train_metrics[k].append(v)
                    if len(train_metrics["train/loss"]) >= self.log_train_updates:
                        self.write_metrics(
                            {k: sum(v) / len(v) for k, v in train_metrics.items()})
                        train_metrics.clear()

                if self.steps % self.update_target_steps == 0:
                    self.update_target_model()

                if self.steps % (self.update_target_steps * 10) == 0:
                    avg_score = sum(score) / len(score) if len(score) > 0 else 0
                    logger.info(
                        "Average episode score: %.3f", avg_score
                    )

                if self.steps % self.eval_steps == 0:
                    eval_metrics = evaluate(
                        self.policy, self.eval_episodes, self.eval_max_steps)
                    self.write_metrics(eval_metrics)
        if self.summary_writer is not None:
            self.summary_writer.close()

    def write_metrics(self, metric_dict: dict):
        """Writes scalar metrics to the summary writer at the current step.

        The writer flushes to disk on its own schedule and on ``close``.

        Args:
            metric_dict: Maps TensorBoard tags to scalar values.
        """
        for k, v in metric_dict.items():
            self.summary_writer.add_scalar(k, v, self.steps)

    def store_experience(self, state, action, reward, next_state, done):
        """Adds one transition to the replay buffer.

        The oldest transition is dropped once the buffer is full.

        Args:
            state: The encoded state before the action, shape ``(N, N, 7)``.
            action: The action index that was taken.
            reward: The reward received for the action.
            next_state: The encoded state after the action.
            done: Whether the episode ended with this action.
        """
        # state is a (n, n, 7) tensor; same as next_state
        # action is a single int
        # reward is a single int
        # done is boolean
        self.experience_replay.append((state, action, reward, next_state, done))

    def sample_experience(self):
        """Draws a random batch of transitions without replacement.

        Returns:
            A list of up to ``batch_size`` transition tuples.
        """
        indices = torch.randperm(len(self.experience_replay))[:self.batch_size].tolist()
        batch = [self.experience_replay[i] for i in indices]
        return batch

    def train_from_experience(self) -> dict:
        """Runs one Double DQN gradient update on a sampled batch.

        The target for each transition is
        ``reward + gamma * Q_target(next_state, argmax Q_policy(next_state))``,
        with the future term dropped for terminal transitions. The argmax
        skips the action that would reverse the snake in ``next_state``
        (see ``Policy.reversal_mask``), the same as the greedy choice.

        Returns:
            A dict of ``train/`` metrics for this update: ``loss``,
            ``grad_norm`` (the total gradient norm before the optimizer
            step), ``q_mean`` and ``q_max`` (of the predicted Q-values of the
            taken actions) and ``target_q_mean``.
        """
        batch = self.sample_experience()
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)

        states = torch.stack(states).to(device=self.device)
        next_states = torch.stack(next_states).to(device=self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Implement Double DQN
        # Use the main model to select the next actions
        with torch.no_grad():
            q_next = self.policy(next_states)

            # mask reversal move
            next_actions = q_next.masked_fill(
                Policy.reversal_mask(next_states), float("-inf")).argmax(dim=-1)

            next_q_values_target = self.target_model(next_states)
            rows = torch.arange(next_states.shape[0], device=next_states.device)
            max_next_q_values = next_q_values_target[rows, next_actions]

        # Compute the target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        q_values = self.policy(states)
        predicted_q_values = q_values[rows, actions]

        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q_values, target_q_values)
        loss.backward()
        # An infinite max norm only measures the norm, it doesn't clip
        grad_norm = clip_grad_norm_(self.policy.parameters(), float("inf"))

        metrics_dict = {
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/q_mean": predicted_q_values.mean().item(),
            "train/q_max": predicted_q_values.max().item(),
            "train/target_q_mean": target_q_values.mean().item(),
        }

        self.optimizer.step()
        return metrics_dict
