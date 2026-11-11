import copy
import logging
import random

from snake_ai.utils import TileMap

logger = logging.getLogger(__name__)


class SnakeEnv:
    def __init__(self):
        self.snake_position = [[0, 0]]
        self.last_snake_position = self.snake_position[0]
        self.direction = None
        self.prev_direction = []
        self.lost = False
        self.won = False
        self.score = 0

        self.price_position = [
            random.randint(0, TileMap.TILE_COUNT.value - 1),
            random.randint(0, TileMap.TILE_COUNT.value - 1)
        ]

        self.reward_decay = 0.995
        self.min_reward = 0.1

    def reset(self):
        self.snake_position = [[0, 0]]
        self.last_snake_position = self.snake_position[0]
        self.direction = None
        self.lost = False
        self.won = False
        self.score = 0
        self.prev_direction = []

        self.price_position = [
            random.randint(0, TileMap.TILE_COUNT.value - 1),
            random.randint(0, TileMap.TILE_COUNT.value - 1)
        ]

        return self.get_state()

    def is_price_collision(self):
        return (self.snake_position[0][0] == self.price_position[0] and
                self.snake_position[0][1] == self.price_position[1])

    def is_snake_collision(self):
        return self.snake_position[0] in self.snake_position[1:]

    def game_over(self):
        self.lost = True

    def game_won(self) -> bool:
        tile_count = TileMap.TILE_COUNT.value * TileMap.TILE_COUNT.value
        return len(self.snake_position) == tile_count

    def handle_collision(self) -> bool:
        # add new square to snake
        self.snake_position.append(self.last_snake_position)
        if self.game_won():
            self.won = True
            return True

        # generate new random location of price
        new_pricex = random.randint(0, TileMap.TILE_COUNT.value - 1)
        new_pricey = random.randint(0, TileMap.TILE_COUNT.value - 1)
        while [new_pricex, new_pricey] in self.snake_position:
            new_pricex = random.randint(0, TileMap.TILE_COUNT.value - 1)
            new_pricey = random.randint(0, TileMap.TILE_COUNT.value - 1)
        self.price_position = [new_pricex, new_pricey]
        return False

    def get_state(self) -> tuple[list[list[int]], list[int]]:
        return self.snake_position, self.price_position

    @staticmethod
    def get_wrapped_distance(pos1, pos2):
        grid_size = TileMap.TILE_COUNT.value
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        dx = min(dx, grid_size - dx)
        dy = min(dy, grid_size - dy)
        return (dx * dx + dy * dy) ** 0.5

    def perform_move(self, direction):
        # Check for invalid move (reversing direction)
        if (
            self.direction == "DOWN" and direction == "UP" or
            self.direction == "UP" and direction == "DOWN" or
            self.direction == "LEFT" and direction == "RIGHT" or
            self.direction == "RIGHT" and direction == "LEFT"
        ):
            logger.debug("Move into direction %s invalid", direction)
            self.game_over()
            return -10, self.get_state(), self.lost, self.won

        self.last_snake_position = copy.copy(self.snake_position[-1])

        # Previous head position
        prev_head_pos = copy.deepcopy(self.snake_position[0])

        # Update the snake's body
        for i in range(len(self.snake_position) - 1, 0, -1):
            self.snake_position[i] = copy.deepcopy(self.snake_position[i - 1])

        # Update snake's head position
        self.direction = direction
        head_pos = self.snake_position[0]

        if direction == "DOWN":
            head_pos[1] = (head_pos[1] + 1) % TileMap.TILE_COUNT.value
        elif direction == "UP":
            head_pos[1] = (head_pos[1] - 1) % TileMap.TILE_COUNT.value
        elif direction == "LEFT":
            head_pos[0] = (head_pos[0] - 1) % TileMap.TILE_COUNT.value
        elif direction == "RIGHT":
            head_pos[0] = (head_pos[0] + 1) % TileMap.TILE_COUNT.value
        else:
            raise Exception(f"Invalid direction detected {direction}")

        self.prev_direction.append(direction)

        # Calculate Euclidean distances to the treasure
        prev_distance = self.get_wrapped_distance(prev_head_pos, self.price_position)
        new_distance = self.get_wrapped_distance(head_pos, self.price_position)

        # Initialize reward
        reward = 0

        # Check collision with treasure
        if self.is_price_collision():
            if self.handle_collision():
                reward = 100
                return reward, self.get_state, self.lost, self.won
            self.score += 1
            reward += 20  # Increased reward for finding treasure
            return reward, self.get_state(), self.lost, self.won

        # Check collision with snake (self-collision)
        if self.is_snake_collision():
            self.game_over()
            reward -= 20  # Increased penalty for collision
            return reward, self.get_state(), self.lost, self.won

        # Reward for moving closer to the treasure
        if new_distance < prev_distance:
            reward += 1  # Small positive reward
        else:
            reward -= 1  # Small negative reward

        # Small penalty per step to encourage efficiency
        reward -= 0.1

        # Add logging
        # print(f"Action: {direction}, Reward: {reward}, \
        # Position: {self.snake_position[0]}, Treasure: {self.price_position}")

        return reward, self.get_state(), self.lost, self.won
