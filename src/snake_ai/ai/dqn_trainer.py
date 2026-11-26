"""Double DQN training for a ``Policy``."""

import copy
import logging
import time
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
from snake_ai.utils import TILE_COUNT

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
        cap: The capacity of the replay buffer, in transitions. Once it is
            full, the oldest transition is overwritten.
        states: The replay buffer's encoded states before each action, a
            ``uint8`` tensor of shape ``(cap, N, N, 7)`` on ``device``. The
            encoded states only hold 0s and 1s, so ``uint8`` is lossless.
        next_states: The encoded states after each action, laid out like
            ``states``.
        actions: The action index of each transition, a ``long`` tensor of
            shape ``(cap,)`` on ``device``.
        rewards: The reward of each transition, a ``float32`` tensor of
            shape ``(cap,)`` on ``device``.
        dones: 1.0 where the transition ended the episode, otherwise 0.0, a
            ``float32`` tensor of shape ``(cap,)`` on ``device``.
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
            ``evaluate`` and a checkpoint is saved.
        eval_episodes: The number of games per evaluation.
        eval_max_steps: The step cap per evaluation game.
        steps: The number of environment steps taken so far.
        device: The device of ``policy``.
        log_train_updates: The number of training updates whose metrics are
            averaged before they are written. ``train/grad_norm`` is the
            exception: it is measured only on the last of these updates.
        model_id: The ``_lr<lr>_g<gamma>`` suffix used in the TensorBoard
            run name and in saved model file names.
        summary_writer: The TensorBoard writer that metrics are written to.
            It writes to ``runs/<%b%d_%H-%M-%S>_<host>_lr<lr>_g<gamma>``
            relative to the current working directory.
    """

    def __init__(
        self,
        policy: Policy,
        seed: int = 11,
        lr: float = 0.001,
        warm_up: int = 10_000,
        train_steps: int = 1_000_000,
        gamma: float = 0.99,
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
            gamma: The discount factor for future rewards. Defaults to 0.99.
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

        self.device = self.policy.device
        self.optimizer = Adam(
            self.policy.parameters(), lr=lr)
        self.criterion = SmoothL1Loss()
        self.batch_size = batch_size
        self.gamma = gamma

        # ring buffers
        self.cap = 500_000
        self.states = torch.zeros((self.cap, TILE_COUNT, TILE_COUNT, 7),
                                  dtype=torch.uint8, device=self.device)
        self.next_states = torch.zeros((self.cap, TILE_COUNT, TILE_COUNT, 7),
                                       dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros(self.cap, device=self.device, dtype=torch.long)
        self.rewards = torch.zeros(self.cap, device=self.device)
        self.dones = torch.zeros(self.cap, device=self.device)

        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.epsilon = 1.0
        self.epsilon_decay_steps = 500_000

        self.train_steps = train_steps
        self.warm_up_steps = warm_up

        self.eval_steps = 100_000
        self.eval_episodes = 50
        self.eval_max_steps = 3000

        # Variables for updating the target model periodically
        self.update_target_steps = update_target
        self.train_policy_steps = update_policy
        self.steps = 0

        # Train metrics are averaged over this many updates before writing
        self.log_train_updates = 100

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.model_id = f"_lr{lr}_g{gamma}"
        self.summary_writer = SummaryWriter(comment=self.model_id)

    def train(self, env, model_path: str | Path):
        """Trains the policy and saves its weights.

        Runs ``run_train``, which also saves a checkpoint after every
        evaluation, and then saves the final policy with ``_save_model``.
        The environment is reset afterward.

        Args:
            env: The environment to train on, usually a ``Game``. See
                ``run_train`` for what it must provide.
            model_path: The directory the checkpoints and the trained model
                are saved to.
        """
        # Train the agent if no model is found
        logger.info("Starting agent training")
        self.run_train(env, model_path)

        logger.info("Finished training agent")

        # Save the trained model
        if not self.steps % self.eval_steps == 0:
            # skipped if it was already saved in run_train
            self._save_model(model_path)
        env.reset()

    def _save_model(self, model_path: str | Path):
        """Saves the policy's state dict at the current step.

        The file is ``model_path / "<timestamp>_model_<model_id>_s<steps>.pt"``,
        e.g. ``2026-11-25_14:00:08_model__lr0.0005_g0.99_s500000.pt``, with
        the timestamp formatted as ``%Y-%m-%d_%H:%M:%S``. The directory is
        created if needed. It can be loaded with ``Policy.load_model``.

        Args:
            model_path: The directory the model is saved to.
        """
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        model_file = (model_path /
                      f"{self.timestamp}_model{self.model_id}_s{self.steps}.pt")
        torch.save(self.policy.state_dict(), model_file)

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

    def run_train(self, env, model_path: str | Path):
        """Runs the training loop for ``train_steps`` environment steps.

        Each step plays one epsilon-greedy move, stores the transition,
        decays ``epsilon``, and may run a training update, a target network
        sync and an evaluation. Every evaluation is followed by a checkpoint
        saved with ``_save_model``. Episode metrics are written when an
        episode ends. The training metrics stay on ``device`` and are
        averaged and written every ``log_train_updates`` updates, which is
        the only point where they wait for the GPU. ``train/grad_norm`` is
        measured then, from the gradients of the last update. The episode
        score average is logged every
        ``10 * update_target_steps`` steps. The summary writer is closed at
        the end.

        Args:
            env: The environment to train on, usually a ``Game``. It must
                provide ``reset()``, ``step(action)``,
                ``is_invalid_move(action)``, ``step_count`` and
                ``snake_env`` (for the episode metrics).
            model_path: The directory the checkpoints are saved to.
        """
        state = self.policy.encode_state(env.reset())
        score = []
        episode = 0
        episode_score = 0
        recent_outcomes = deque(maxlen=100)
        train_metrics = defaultdict(list)
        with logging_redirect_tqdm():
            for _ in tqdm(range(self.train_steps), desc="train"):
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
                        "replay/size": min(self.steps, self.cap),
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
                        metrics = {k: torch.stack(v).mean()
                                   for k, v in train_metrics.items()}
                        # An infinite max norm only measures the norm, it doesn't clip
                        grad_norm = clip_grad_norm_(
                            self.policy.parameters(), float("inf")
                        )
                        metrics["train/grad_norm"] = grad_norm
                        metrics = {k: v.item() for k, v in metrics.items()}
                        self.write_metrics(metrics)
                        train_metrics.clear()

                if self.steps % self.update_target_steps == 0:
                    self.update_target_model()

                if self.steps % (self.update_target_steps * 10) == 0:
                    avg_score = sum(score) / len(score) if len(score) > 0 else 0
                    logger.info(
                        "Average episode score: %.3f", avg_score
                    )

                if self.steps % self.eval_steps == 0:
                    eval_start = time.perf_counter()
                    eval_metrics = evaluate(
                        self.policy, self.eval_episodes, self.eval_max_steps,
                        progress=True)
                    logger.info("Evaluation took %.1fs",
                                time.perf_counter() - eval_start)
                    self.write_metrics(eval_metrics)
                    self._save_model(model_path)
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
        """Writes one transition into the replay buffer.

        The slot is ``steps % cap``, so the buffer fills from slot 0 and,
        once full, overwrites the oldest transition. This relies on
        ``run_train`` calling it before it increments ``steps``. The values
        are copied to ``device``, and the states are stored as ``uint8``.

        Args:
            state: The encoded state before the action, shape ``(N, N, 7)``.
            action: The action index that was taken.
            reward: The reward received for the action.
            next_state: The encoded state after the action.
            done: Whether the episode ended with this action.
        """
        next_pos = self.steps % self.cap
        self.states[next_pos] = state
        self.next_states[next_pos] = next_state
        self.actions[next_pos] = action
        self.rewards[next_pos] = reward
        self.dones[next_pos] = done

    def train_from_experience(self) -> dict:
        """Runs one Double DQN gradient update on a sampled batch.

        The target for each transition is
        ``reward + gamma * Q_target(next_state, argmax Q_policy(next_state))``,
        with the future term dropped for terminal transitions. The argmax
        skips the action that would reverse the snake in ``next_state``
        (see ``Policy.reversal_mask``), the same as the greedy choice.

        The batch is drawn uniformly, with replacement, from the filled part
        of the replay buffer, and stays on ``device``.

        Returns:
            A dict of ``train/`` metrics for this update: ``loss``,
            ``q_mean`` and ``q_max`` (of the predicted Q-values of the taken
            actions) and ``target_q_mean``. The values are detached 0-dim
            tensors on ``device``, so building the dict doesn't wait for the
            GPU. ``train/grad_norm`` is not included; ``run_train`` measures
            it when it writes the metrics.
        """

        # sample new batch
        idx = torch.randint(min(self.steps, self.cap), (self.batch_size,),
                            device=self.device)
        states = self.states[idx].float()
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = self.next_states[idx].float()
        dones = self.dones[idx]

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

        metrics_dict = {
            "train/loss": loss.detach(),
            "train/q_mean": predicted_q_values.detach().mean(),
            "train/q_max": predicted_q_values.detach().max(),
            "train/target_q_mean": target_q_values.detach().mean(),
        }

        self.optimizer.step()
        return metrics_dict
