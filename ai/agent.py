import tensorflow as tf
import numpy as np
from snake.utils import Utils
from snake.controller import Controller
from threading import Thread

# This file was created using the following tutorial book:
# Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow by Aurelien Geron


class Agent:
    def __init__(self):
        print("Initializing Agent")

        self.input_space = Utils.TILE_COUNT.value
        self.controller = Controller()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation='relu', name="input"),
            tf.keras.layers.Dense(4, activation='softmax', name="output")
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        print("Training Agent")
        for it in range(100):
            message = f"\rIteration {it+1}/100"
            print(message, end="", sep=" ", flush=True)
            rewards, grads = self.play(1, 100)
            discounted_normalized_rewards = self.discount_and_normalize_rewards(rewards, discount_rate=0.95)

            mean_grads = []
            for var_index in range(len(self.model.trainable_variables)):
                mean_grads.append(
                    tf.reduce_mean(
                        [discounted_normalized_reward * grads[episode_index][step][var_index]
                         for episode_index, rewards in enumerate(discounted_normalized_rewards)
                         for step, discounted_normalized_reward in enumerate(rewards)], axis=0))
            self.optimizer.apply_gradients(zip(mean_grads, self.model.trainable_variables))

        print("Agent Trained")

    def play_one_step(self, state):
        with tf.GradientTape() as tape:
            q_values = self.model(state)
            # set the index of the highest value to 1 and the rest to 0
            action = tf.one_hot(tf.argmax(q_values, axis=1), 4)
            action = tf.cast(action, tf.float32)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(action, q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)

        # make a move based on the q values and get the reward and new state
        reward, new_state = self.controller.perform_move(action)
        return reward, new_state, grads

    def play(self, n_episodes, n_max_steps):
        rewards = []
        grads = []
        for episode in range(n_episodes):
            current_rewards = []
            current_grads = []
            state = self.controller.model.get_state()
            for step in range(n_max_steps):
                state = np.array(state).reshape(1, self.input_space * self.input_space)
                reward, state, grad = self.play_one_step(state)
                current_rewards.append(reward)
                current_grads.append(grad)
                if self.controller.model.lost:
                    break
            rewards.append(current_rewards)
            grads.append(current_grads)
        return rewards, grads

    @staticmethod
    def discount_rewards(rewards, discount_rate):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_rate
        return discounted

    def discount_and_normalize_rewards(self, rewards, discount_rate):
        discounted_rewards = [self.discount_rewards(reward, discount_rate) for reward in rewards]
        flat_rewards = np.concatenate(discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_reward - reward_mean) / reward_std for discounted_reward in discounted_rewards]
