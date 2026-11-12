"""The Double DQN agent that learns and plays Snake."""

import logging
import random
from collections import deque

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from snake_ai.utils import TileMap

from .policy_model import PolicyModel

logger = logging.getLogger(__name__)


class Agent:
    """A Double DQN agent with experience replay and epsilon-greedy exploration.

    The agent holds two ``PolicyModel`` networks. The policy network is
    trained and picks actions. The target network is a copy that is synced
    every ``update_target_steps`` steps and is used to compute stable
    training targets. Following Double DQN, the policy network chooses the
    next action and the target network scores it.

    For serving, call ``load_model`` once and then ``act`` for each move.

    Attributes:
        policy_model: The network that is trained and used to pick actions.
        target_model: A periodically synced copy used for training targets.
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
        device: ``"cuda"`` if available, otherwise ``"cpu"``.
    """

    def __init__(self):
        """Builds both networks and sets the training hyperparameters.

        The networks start with random weights. Call ``load_model`` or
        ``train_or_load_model`` to get a trained agent.
        """
        print("Initializing Agent")

        self.input_space = TileMap.TILE_COUNT.value
        self.policy_model = PolicyModel(3)
        self.target_model = PolicyModel(3)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.optimizer = Adam(self.policy_model.parameters(), lr=0.001)
        self.criterion = MSELoss(reduction="mean")
        self.experience_replay = deque(maxlen=10000)
        self.batch_size = 256
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.gamma = 0.95
        
        self.train_steps = 1_000_000
        self.learning_starts = 10_000

        # Variables for updating the target model periodically
        self.update_target_steps = 1000
        self.train_policy_steps = 4
        self.steps = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            print(f"Warning: Using {self.device} to train the model")
        self.policy_model.to(self.device)
        self.target_model.to(self.device)

    def train_or_load_model(self, env):
        """Loads the saved model, or trains a new one if none can be loaded.

        A newly trained model is saved to ``policy_models/model.pt`` and the
        environment is reset afterwards.

        Args:
            env: The environment to train on. It must provide ``reset()``
                and ``step(action)``.
        """
        # Try to load an existing model
        try:
            self.policy_model.load_state_dict(
                torch.load("policy_models/model.pt"))
            self.update_target_model()
            print("Model Loaded")
            return
        except Exception:
            print("No Model Found")

        # Train the agent if no model is found
        print("Training Agent")
        self.train(env)

        print("\nAgent Trained")

        # Save the trained model
        torch.save(self.policy_model.state_dict(), "policy_models/model.pt")
        env.reset()

    def load_model(self):
        """Loads the policy weights from ``policy_models/model.pt``.

        The path is relative to the current working directory. The target
        network is synced afterwards. If loading fails for any reason, the
        error is swallowed and the agent keeps its random weights.
        """
        # Try to load an existing model
        try:
            self.policy_model.load_state_dict(
                torch.load("policy_models/model.pt"))
            self.update_target_model()
            print("Model Loaded")
            return
        except Exception:
            print("No Model Found")

    def update_target_model(self):
        """Copies the policy network's weights into the target network."""
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def play_one_step(self, state, env):
        """Picks an epsilon-greedy action and applies it to the environment.

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
            action = self.get_move(state)

        reward, new_state, done = env.step(action)
        new_state = self.encode_state(new_state)

        return reward, new_state, action, done

    def train(self, env):
        """Runs the training loop for ``train_steps`` environment steps.

        Each step plays one epsilon-greedy move, stores the transition, and
        may run training updates and a target network sync. The episode
        score average is logged every ``10 * update_target_steps`` steps.

        Args:
            env: The environment to train on. It must provide ``reset()``
                and ``step(action)``.
        """
        state = self.encode_state(env.reset())
        avg_score = []
        episode_score = 0
        with logging_redirect_tqdm():
            for _ in tqdm(range(self.train_steps)):
                # Decay epsilon after each episode
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                reward, next_state, action, done = self.play_one_step(state, env)

                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                self.steps += 1
                episode_score += reward

                if done:
                    state = self.encode_state(env.reset())
                    avg_score.append(episode_score)
                    episode_score = 0

                if (
                    self.steps > self.learning_starts and
                    self.steps % self.train_policy_steps == 0
                ):
                    self.train_from_experience()

                if self.steps % self.update_target_steps == 0:
                    self.update_target_model()

                if len(self.experience_replay) >= self.batch_size:
                    self.train_from_experience()

                if self.steps % (self.update_target_steps * 10) == 0:
                    logger.info(
                        "Average episode score: %.3f",
                        sum(avg_score) / len(avg_score))

        # print(
        #     f"Episode {episode + 1}/{n_episodes}, "
        #     f"Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}"
        # )

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
            next_actions = self.policy_model(next_states).argmax(dim=-1)
            next_q_values_target = self.target_model(next_states)
            rows = torch.arange(next_states.shape[0], device=next_states.device)
            max_next_q_values = next_q_values_target[rows, next_actions]

        # Compute the target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        q_values = self.policy_model(states)
        predicted_q_values = q_values[rows, actions]

        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q_values, target_q_values)
        loss.backward()

        self.optimizer.step()

    def get_move(self, state):
        """Returns the greedy action for an encoded state.

        Args:
            state: An encoded state of shape ``(N, N, 3)``.

        Returns:
            The index of the action with the highest Q-value.
        """
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.policy_model(state)
        action = torch.argmax(q_values, dim=-1).item()
        return action

    def act(self, plain_state):
        """Returns the greedy action for a plain game state.

        This is the inference entry point the server uses.

        Args:
            plain_state: A ``(snake_position, price_position)`` tuple, as
                returned by ``SnakeEnv.get_state``.

        Returns:
            An action index into ``snake.game.ACTIONS``.
        """
        return self.get_move(self.encode_state(plain_state))

    @staticmethod
    def encode_state(plain_state):
        """Turns a plain game state into the network's input tensor.

        Args:
            plain_state: A ``(snake_position, price_position)`` tuple. The
                snake positions are ``[x, y]`` pairs, head first.

        Returns:
            A float tensor of shape ``(N, N, 3)``, indexed as ``[x, y, c]``.
            Channel 0 marks the head, channel 1 the rest of the body and
            channel 2 the food.
        """
        snake_position, price_position = plain_state
        grid_size = TileMap.TILE_COUNT.value

        # Initialize the state grid with zeros for each channel
        # We will use 3 channels:
        # Channel 0: Snake's head
        # Channel 1: Snake's body (excluding head)
        # Channel 2: Treasure
        state = torch.zeros(grid_size, grid_size, 3, dtype=torch.float32)

        # Channel 0: Mark the snake's head position
        head_pos = snake_position[0]
        state[head_pos[0], head_pos[1], 0] = 1.0

        # Channel 1: Mark the snake's body positions (excluding head)
        for pos in snake_position[1:]:
            state[pos[0], pos[1], 1] = 1.0

        # Channel 2: Mark the treasure position
        state[price_position[0], price_position[1], 2] = 1.0

        return state
