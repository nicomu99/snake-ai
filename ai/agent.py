import tensorflow as tf
import numpy as np
from snake.utils import Utils
import time
from tqdm import tqdm
from collections import deque


class Agent:
    def __init__(self):
        print("Initializing Agent")

        self.input_space = Utils.TILE_COUNT.value
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                   input_shape=(self.input_space, self.input_space, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')  # Linear activation for Q-values
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.experience_replay = deque(maxlen=10000)  # Use deque for efficient buffer management
        self.batch_size = 64
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.99  # Decay rate for exploration probability
        self.epsilon_min = 0.1  # Minimum exploration probability
        self.gamma = 0.95

        # Initialize the target model
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Variables for updating the target model periodically
        self.update_target_steps = 1000  # Update target model every 1000 steps
        self.steps = 0  # Counter for steps taken

    def train_or_load_model(self, controller):
        # Try to load an existing model
        try:
            self.model = tf.keras.models.load_model("model.keras")
            # Load the weights into the target model as well
            self.target_model.set_weights(self.model.get_weights())
            print("Model Loaded")
            return
        except Exception:
            print("No Model Found")

        # Train the agent if no model is found
        print("Training Agent")
        for _ in tqdm(range(40)):
            self.play_multiple_episodes(40, 500, controller)

        print("\nAgent Trained")

        # Save the trained model
        self.model.save("model.keras")

        controller.reset()

    def play_one_step(self, state, controller):
        # Ensure state has the correct shape
        state = np.expand_dims(state, axis=0)  # Shape: (1, grid_size, grid_size, 3)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(4)  # Explore: select a random action
        else:
            q_values = self.model(state)
            action = tf.argmax(q_values, axis=1).numpy()[0]  # Exploit: select the action with max Q-value

        reward, new_state, done = controller.perform_move(action)

        return reward, new_state, action, done

    def play_multiple_episodes(self, n_episodes, n_max_steps, controller):
        for episode in range(n_episodes):
            state = controller.reset()
            total_reward = 0
            for step in range(n_max_steps):
                reward, next_state, action, done = self.play_one_step(state, controller)
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.steps += 1  # Increment step counter

                # Train the model if enough experiences are gathered
                if len(self.experience_replay) >= self.batch_size:
                    self.train_from_experience()

                if done:
                    break

            # Decay epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

    def store_experience(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def sample_experience(self):
        indices = np.random.choice(len(self.experience_replay), size=self.batch_size, replace=False)
        batch = [self.experience_replay[i] for i in indices]
        return batch

    def update_target_model(self):
        # Copy weights from the main model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def train_from_experience(self):
        batch = self.sample_experience()
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors with correct data types
        states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        # Implement Double DQN
        # Use the main model to select the next actions
        next_actions = tf.argmax(self.model(next_states), axis=1)
        next_actions = tf.cast(next_actions, tf.int32)

        # Use the target model to evaluate the Q-values of the next actions
        next_q_values_target = self.target_model(next_states)
        indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), next_actions], axis=1)
        max_next_q_values = tf.gather_nd(next_q_values_target, indices)

        # Compute the target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        with tf.GradientTape() as tape:
            # Get the predicted Q-values for the current states and actions
            q_values = self.model(states)
            indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), actions], axis=1)
            predicted_q_values = tf.gather_nd(q_values, indices)

            # Compute the loss between target and predicted Q-values
            loss = tf.reduce_mean(tf.square(target_q_values - predicted_q_values))

        # Compute gradients and update the main model's weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update the target model periodically
        if self.steps % self.update_target_steps == 0:
            self.update_target_model()

    def get_move(self, state):
        # Ensure state has the correct shape
        state = np.expand_dims(state, axis=0)
        q_values = self.model(state)
        action = tf.argmax(q_values, axis=1).numpy()[0]
        return action

    def play_steps(self, controller):
        state = controller.model.get_state()
        for i in range(5):
            action = self.get_move(state)
            reward, state, _ = controller.perform_move(action)
            time.sleep(0.1)
