"""The Snake game rules, including reward shaping for training."""

import copy
import logging
import random

from snake_ai.utils import TILE_COUNT

logger = logging.getLogger(__name__)


class SnakeEnv:
    """The raw state and rules of a Snake game.

    The board is ``TILE_COUNT`` tiles on each side and wraps at the
    edges, so it is a torus and there are no walls. The snake starts as a
    single tile at ``[0, 0]`` with no direction. Positions are ``[x, y]``
    lists, where ``y`` grows downwards. "Price" means the food throughout.

    The game is lost when the snake runs into itself or reverses its
    direction, and won when the snake fills the whole board.

    Attributes:
        snake_position: The snake's tiles as ``[x, y]`` lists, head first.
        last_snake_position: The tail position before the last move. The
            snake grows into it after eating.
        direction: The direction of the last move, or ``None`` before the
            first move.
        lost: Whether the game is lost.
        won: Whether the game is won.
        score: The amount of food eaten.
        price_position: The food's ``[x, y]`` position.
    """

    def __init__(self):
        self.snake_position = None
        self.last_snake_position = None
        self.direction = None
        self.lost = False
        self.won = False
        self.score = 0
        self.price_position = None

        self.reset()

    def reset(self):
        """Starts a new game with a one-tile snake and new food.

        Returns:
            The plain state of the new game, see ``get_state``.
        """
        self.snake_position = [[0, 0]]
        self.last_snake_position = self.snake_position[0]
        self.direction = None
        self.lost = False
        self.won = False
        self.score = 0

        self._generate_price()

        return self.get_state()

    def is_price_collision(self):
        """Returns whether the snake's head is on the food."""
        return (self.snake_position[0][0] == self.price_position[0] and
                self.snake_position[0][1] == self.price_position[1])

    def is_snake_collision(self):
        """Returns whether the snake's head overlaps its body."""
        return self.snake_position[0] in self.snake_position[1:]

    def game_over(self):
        """Marks the game as lost."""
        self.lost = True

    def game_won(self) -> bool:
        """Returns whether the snake fills the whole board."""
        tile_count = TILE_COUNT * TILE_COUNT
        return len(self.snake_position) == tile_count

    def handle_collision(self) -> bool:
        """Grows the snake after eating and places new food.

        The snake grows by one tile at its previous tail position. If it now
        fills the board, the game is marked as won and no new food is placed.
        Otherwise, the food moves to a random free tile.

        Returns:
            True if the game is now won, otherwise False.
        """
        # add new square to snake
        self.snake_position.append(self.last_snake_position)
        if self.game_won():
            self.won = True
            return True

        self._generate_price()
        return False
    
    def _generate_price(self):
        new_pricex = random.randint(0, TILE_COUNT - 1)
        new_pricey = random.randint(0, TILE_COUNT - 1)
        while [new_pricex, new_pricey] in self.snake_position:
            new_pricex = random.randint(0, TILE_COUNT - 1)
            new_pricey = random.randint(0, TILE_COUNT - 1)
        self.price_position = [new_pricex, new_pricey]

    def get_state(self) -> tuple[list[list[int]], list[int]]:
        """Returns the plain game state.

        The returned lists are the env's own lists, not copies, so they
        change with the next move.

        Returns:
            A ``(snake_position, price_position)`` tuple.
        """
        return copy.deepcopy(self.snake_position), copy.deepcopy(self.price_position)

    @staticmethod
    def get_wrapped_distance(pos1, pos2):
        """Returns the Euclidean distance between two tiles on the torus.

        On each axis, the shorter way around the board is used.

        Args:
            pos1: The first ``[x, y]`` position.
            pos2: The second ``[x, y]`` position.

        Returns:
            The distance as a float.
        """
        grid_size = TILE_COUNT
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        dx = min(dx, grid_size - dx)
        dy = min(dy, grid_size - dy)
        return (dx * dx + dy * dy) ** 0.5

    def perform_move(self, direction):
        """Moves the snake one tile and returns the shaped reward.

        The checks run in this order:

        1. Reversing the current direction loses the game (reward -10). The
           snake does not move.
        2. Eating the food grows the snake: +100 if that wins the game,
           otherwise +20 and the score goes up by one.
        3. Running into its own body loses the game (reward -20).
        4. Otherwise +1 if the head got closer to the food (wrapped distance),
           -1 if not, and -0.1 per step on top.

        Args:
            direction: ``"UP"``, ``"DOWN"``, ``"LEFT"`` or ``"RIGHT"``.

        Returns:
            A ``(reward, state, lost, won)`` tuple, where ``state`` is the
            result of ``get_state``.

        Raises:
            ValueError: If ``direction`` is not one of the four directions.
        """
        if self.won or self.lost:
            return 0, self.get_state(), self.lost, self.won
        
        if direction not in ["DOWN", "UP", "LEFT", "RIGFHT"]:
            raise ValueError(f"Invalid direction detected {direction}")
        
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
            head_pos[1] = (head_pos[1] + 1) % TILE_COUNT
        elif direction == "UP":
            head_pos[1] = (head_pos[1] - 1) % TILE_COUNT
        elif direction == "LEFT":
            head_pos[0] = (head_pos[0] - 1) % TILE_COUNT
        elif direction == "RIGHT":
            head_pos[0] = (head_pos[0] + 1) % TILE_COUNT

        # Calculate Euclidean distances to the treasure
        prev_distance = self.get_wrapped_distance(prev_head_pos, self.price_position)
        new_distance = self.get_wrapped_distance(head_pos, self.price_position)

        # Initialize reward
        reward = 0

        # Check collision with treasure
        if self.is_price_collision():
            self.score += 1
            if self.handle_collision():
                reward = 100
                return reward, self.get_state(), self.lost, self.won
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

        return reward, self.get_state(), self.lost, self.won
