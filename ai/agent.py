import tensorflow as tf
from snake.utils import Utils
from snake.controller import Controller


class Agent:
    def __init__(self):
        # this is the agent implementing reiforcement learning
        # I need: a reward
        # Each move we put in the current state and let the neural net decide where to go
        # if the neural network makes a mistake we have to

        # action space: 4 actions, UP DOWN LEFT RIGHT
        # state space: 10x10 grid, 100 tiles in total
        # reward: 1 for eating, -1 for dying, 0 for moving

        self.input_space = Utils.TILE_COUNT.value
        self.controller = Controller()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.input_space, self.input_space)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4)
        ])

        pass

    def play_one_step(self, state):
        with tf.GradientTape() as tape:
            q_values = self.model(state)
            action = tf.argmax(q_values, axis=1)
            loss = tf.reduce_mean(tf.square(action - q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)

        # make a move based on the q values and get the reward and new state
        reward, new_state = self.make_move(action)
        return reward, new_state, grads

    def make_move(self, action):
        # make a move based on the action and return the reward and new state
        pass
