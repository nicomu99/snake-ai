"""Double DQN training for a ``Policy``."""

import copy
import logging
import random
from collections import deque
from datetime import datetime

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from snake_ai.utils import MODEL_BASE_PATH

from .policy import Policy

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
        policy: The policy that is trained and used to pick actions. It is
            updated in place.
        target_model: A frozen, periodically synced copy of ``policy`` used
            for training targets.
        optimizer: Adam optimizer for the policy network.
        criterion: MSE loss between predicted and target Q-values.
        experience_replay: A bounded buffer of
            ``(state, action, reward, next_state, done)`` transitions.
        batch_size: The number of transitions per training update.
        epsilon: The current probability of taking a random action.
        epsilon_decay: The factor ``epsilon`` is multiplied by each step.
        epsilon_min: The lower bound for ``epsilon``.
        gamma: The discount factor for future rewards.
        train_steps: The total number of environment steps in ``train``.
        learning_starts: Steps to collect before the gated training updates
            begin.
        update_target_steps: How often (in steps) the target network is synced.
        train_policy_steps: How often (in steps) the gated training update
            runs once ``learning_starts`` is reached.
        steps: The number of environment steps taken so far.
        device: The device of ``policy``.
    """

    def __init__(self, policy: Policy):
        """Sets up the target network, optimizer and training hyperparameters.

        Args:
            policy: The policy to train. It can be freshly initialized or
                already loaded with weights to continue training. The target
                network starts as a copy of it.
        """
        logger.info("Initializing agent")

        self.policy = policy
        self.target_model = copy.deepcopy(self.policy)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        self.optimizer = Adam(
            self.policy.parameters(), lr=0.001)
        self.criterion = MSELoss(reduction="mean")
        self.experience_replay = deque(maxlen=10000)
        self.batch_size = 256
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.gamma = 0.95
        
        self.train_steps = 200_000
        self.learning_starts = 10_000

        # Variables for updating the target model periodically
        self.update_target_steps = 1000
        self.train_policy_steps = 4
        self.steps = 0

        self.device = self.policy.device

    def train(self, env):
        """Trains the policy and saves its weights.

        Runs ``run_train`` and then saves the policy's state dict to
        ``MODEL_BASE_PATH / "<timestamp>_model_<train_steps>.pt"``, with the
        timestamp formatted as ``%Y-%m-%d_%H:%M:%S``. The directory is
        created if needed. The environment is reset afterward.

        Args:
            env: The environment to train on. It must provide ``reset()``
                and ``step(action)``.
        """
        # Train the agent if no model is found
        logger.info("Starting agent training")
        self.run_train(env)

        logger.info("Finished training agent")

        # Save the trained model
        MODEL_BASE_PATH.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_file = MODEL_BASE_PATH / f"{timestamp}_model_{self.train_steps}.pt"
        torch.save(self.policy.state_dict(), model_file)
        env.reset()

    def update_target_model(self):
        """Copies the policy network's weights into the target network."""
        self.target_model.load_state_dict(self.policy.state_dict())

    def play_one_step(self, state, env):
        """Picks an epsilon-greedy action and applies it to the environment.

        With probability ``epsilon`` a random action is taken, otherwise the
        policy's greedy action.

        Args:
            state: The current encoded state.
            env: The environment to step.

        Returns:
            A ``(reward, new_state, action, done)`` tuple, where ``new_state``
            is already encoded.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.policy.get_move(state)

        reward, new_state, lost, won = env.step(action)
        new_state = self.policy.encode_state(new_state)

        return reward, new_state, action, lost or won

    def run_train(self, env):
        """Runs the training loop for ``train_steps`` environment steps.

        Each step plays one epsilon-greedy move, stores the transition, and
        may run training updates and a target network sync. The episode
        score average is logged every ``10 * update_target_steps`` steps.

        Args:
            env: The environment to train on. It must provide ``reset()``
                and ``step(action)``.
        """
        state = self.policy.encode_state(env.reset())
        score = []
        episode_score = 0
        with ((logging_redirect_tqdm())):
            for _ in tqdm(range(self.train_steps)):
                reward, next_state, action, done = self.play_one_step(state, env)

                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                self.steps += 1
                episode_score += reward

                if done:
                    state = self.policy.encode_state(env.reset())
                    score.append(episode_score)
                    episode_score = 0

                    # Decay epsilon after each episode
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

                if (
                    self.steps > self.learning_starts and
                    self.steps % self.train_policy_steps == 0
                ):
                    self.train_from_experience()

                if self.steps % self.update_target_steps == 0:
                    self.update_target_model()

                if self.steps % (self.update_target_steps * 10) == 0:
                    avg_score = sum(score) / len(score) if len(score) > 0 else 0
                    logger.info(
                        "Average episode score: %.3f", avg_score
                    )

    def store_experience(self, state, action, reward, next_state, done):
        """Adds one transition to the replay buffer.

        The oldest transition is dropped once the buffer is full.

        Args:
            state: The encoded state before the action, shape ``(N, N, 3)``.
            action: The action index that was taken.
            reward: The reward received for the action.
            next_state: The encoded state after the action.
            done: Whether the episode ended with this action.
        """
        # state is a (n, n, 3) tensor; same as next_state
        # action is a single int
        # reward is a single int
        # done is boolean
        self.experience_replay.append((state, action, reward, next_state, done))

    def sample_experience(self):
        """Draws a random batch of transitions without replacement.

        Returns:
            A list of up to ``batch_size`` transition tuples.
        """
        indices = torch.randperm(len(self.experience_replay))[:self.batch_size]
        batch = [self.experience_replay[i] for i in indices]
        return batch

    def train_from_experience(self):
        """Runs one Double DQN gradient update on a sampled batch.

        The target for each transition is
        ``reward + gamma * Q_target(next_state, argmax Q_policy(next_state))``,
        with the future term dropped for terminal transitions.
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
            next_actions = self.policy(next_states).argmax(dim=-1)
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

        self.optimizer.step()
