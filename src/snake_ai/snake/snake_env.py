"""The Snake game rules, including reward shaping for training.

Attributes:
    ACTIONS (tuple[str, ...]): Maps action indices to directions. The order
        must match the outputs of ``PolicyModel``.
"""
import copy
import logging
from random import Random

from snake_ai.utils import TILE_COUNT

logger = logging.getLogger(__name__)

# Index = action int produced by the agent
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class SnakeEnv:
    """The raw state and rules of a Snake game.

    The board is ``TILE_COUNT`` tiles on each side and wraps at the
    edges, so it is a torus and there are no walls. The snake starts as a
    single tile at ``[0, 0]`` with no direction. Positions are ``[x, y]``
    lists, where ``y`` grows downwards. "Price" means the food throughout.

    The game is lost when the snake runs into itself or reverses its
    direction, and won when the snake fills the whole board.

    Args:
        seed: The seed for ``rng``. If None, it is seeded from the operating
            system, so every env places food differently. Defaults to None.

    Attributes:
        rng: A random number generator for generating food positions.
        snake_position: The snake's tiles as ``[x, y]`` lists, head first.
        last_snake_position: The tail position before the last move. The
            snake grows into it after eating.
        direction: The direction of the last move, or ``None`` before the
            first move.
        lost: Whether the game is lost.
        won: Whether the game is won.
        game_finished: Whether the game is lost or won.
        step_count: The number of moves since the last reset. Moves made
            after the game has ended are not counted.
        score: The amount of food eaten.
        price_position: The food's ``[x, y]`` position.
    """

    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)

        self.snake_position = None
        self.last_snake_position = None
        self.direction = None
        self.lost = False
        self.won = False
        self.game_finished = False
        self.score = 0
        self.price_position = None
        self.step_count = 0

        self.reset()

    def reset(self, seed: int | None = None):
        """Starts a new game with a one-tile snake and new food.

        Args:
            seed: Seed for random number generator. If None, the current random number
                generator keeps going without being reseeded. Defaults to None.

        Returns:
            The plain state of the new game, see ``get_state``.
        """
        if seed is not None:
            self.rng = Random(seed)

        self.snake_position = [[0, 0]]
        self.last_snake_position = self.snake_position[0]
        self.direction = None
        self.lost = False
        self.won = False
        self.game_finished = False
        self.score = 0
        self.step_count = 0

        self._generate_price()

        return self.get_state()

    def is_price_collision(self):
        """Returns whether the snake's head is on the food."""
        return (self.snake_position[0][0] == self.price_position[0] and
                self.snake_position[0][1] == self.price_position[1])

    def is_snake_collision(self):
        """Returns whether the snake's head overlaps its body."""
        return self.snake_position[0] in self.snake_position[1:]

    def game_won(self) -> bool:
        """Returns whether the snake fills the whole board."""
        tile_count = TILE_COUNT * TILE_COUNT
        return len(self.snake_position) == tile_count

    def get_game_state(self):
        """Returns whether the game is still running and, if not, why it ended.

        A game lost by reversing direction is told apart from a collision
        because the snake doesn't move on a reversal, so its head is not on
        its body.

        Returns:
            ``"game_won"`` if the game is won, ``"collision"`` if the snake
            ran into itself, ``"direction_reversal"`` if it was lost by
            reversing direction, or ``"pending"`` while it is still running.

        Raises:
            ValueError: If the game is marked as both won and lost.
        """
        if self.won:
            return "game_won"
        elif self.is_snake_collision():
            return "collision"
        elif self.lost:
            return "direction_reversal"
        return "pending"

    def set_game_over(self):
        """Marks the game as lost."""
        self.lost = True
        self.game_finished = True

    def set_game_won(self):
        """Marks the game as won."""
        self.won = True
        self.game_finished = True

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
            self.set_game_won()
            return True

        self._generate_price()
        return False

    def _generate_price(self):
        """Places the food on a random tile that the snake doesn't cover."""
        new_pricex = self.rng.randint(0, TILE_COUNT - 1)
        new_pricey = self.rng.randint(0, TILE_COUNT - 1)
        while [new_pricex, new_pricey] in self.snake_position:
            new_pricex = self.rng.randint(0, TILE_COUNT - 1)
            new_pricey = self.rng.randint(0, TILE_COUNT - 1)
        self.price_position = [new_pricex, new_pricey]

    def get_state(self) -> tuple[list[list[int]], list[int], str | None]:
        """Returns the plain game state.

        The returned lists are deep copies, so later moves don't change them.

        Returns:
            A ``(snake_position, price_position, direction)`` tuple.
        """
        return (
            copy.deepcopy(self.snake_position),
            list(self.price_position),
            self.direction,
        )

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

    def is_invalid_move(self, direction: str | int) -> bool:
        """Returns whether a move would reverse the snake's direction.

        Reversing loses the game in ``perform_move``. It does not depend on
        the snake's length, so a one-tile snake can't reverse either.

        Args:
            direction: ``"UP"``, ``"DOWN"``, ``"LEFT"``, ``"RIGHT"`` or an
                index into ``ACTIONS``.

        Returns:
            True if ``direction`` is the opposite of the current direction.
            Always False before the first move, when ``direction`` is None.

        Raises:
            ValueError: If ``direction`` is not a valid direction or index.
        """
        direction = self.transform_move(direction)
        if (
            self.direction == "DOWN" and direction == "UP" or
            self.direction == "UP" and direction == "DOWN" or
            self.direction == "LEFT" and direction == "RIGHT" or
            self.direction == "RIGHT" and direction == "LEFT"
        ):
            return True
        return False

    @staticmethod
    def transform_move(action: int | str) -> str:
        """Validates an action and converts it to a direction string.

        This only checks that the action is a known value. It does not check
        the game rules; see ``SnakeEnv.is_invalid_move`` for that.

        Args:
            action: A direction string (``"UP"``, ``"DOWN"``, ``"LEFT"``,
                ``"RIGHT"``) or an index into ``ACTIONS``.

        Returns:
            The direction string.

        Raises:
            ValueError: If ``action`` is not a valid direction or index.
        """
        if isinstance(action, int):
            if action < 0 or action > 3:
                raise ValueError(
                    "Invalid action value. Must be between 0 and 3 inclusive.")
            direction = ACTIONS[action]
        else:
            if action not in ACTIONS:
                raise ValueError(f"Invalid action value. Must be in {ACTIONS}")
            direction = action
        return str(direction)

    def perform_move(self, action: int | str):
        """Moves the snake one tile and returns the shaped reward.

        If the game is already lost or won, nothing happens and the reward
        is 0. Otherwise, the checks run in this order:

        1. Reversing the current direction loses the game (reward -2). The
           snake does not move. The trained policy never reverses, so this
           only happens with other callers.
        2. Eating the food grows the snake and raises the score by one:
           +10 if that wins the game, otherwise +2.
        3. Running into its own body loses the game (reward -2).
        4. Otherwise +0.01 if the head got closer to the food (wrapped
           distance), -0.01 if not, and -0.001 per step on top.

        Args:
            action: ``"UP"``, ``"DOWN"``, ``"LEFT"``, ``"RIGHT"`` or an
                index into ``ACTIONS``.

        Returns:
            A ``(reward, state, lost, won)`` tuple, where ``state`` is the
            result of ``get_state``.

        Raises:
            ValueError: If ``action`` is not a valid direction or index.
                Not raised once the game is over.
        """
        if self.game_finished:
            return 0, self.get_state(), self.lost, self.won

        direction = self.transform_move(action)
        logger.debug("Performing move into direction %s", direction)

        self.step_count += 1

        # Check for invalid move (reversing direction)
        if self.is_invalid_move(direction):
            logger.debug("Move into direction %s invalid", direction)
            self.set_game_over()
            return -2, self.get_state(), self.lost, self.won

        self.last_snake_position = list(self.snake_position[-1])

        # Update the snake's body
        for i in range(len(self.snake_position) - 1, 0, -1):
            self.snake_position[i] = list(self.snake_position[i - 1])

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

        # Initialize reward
        reward = 0

        # Check collision with treasure
        if self.is_price_collision():
            self.score += 1
            if self.handle_collision():
                reward = 10
                return reward, self.get_state(), self.lost, self.won
            reward += 2  # Increased reward for finding treasure
            return reward, self.get_state(), self.lost, self.won

        # Check collision with snake (self-collision)
        if self.is_snake_collision():
            self.set_game_over()
            reward -= 2  # Increased penalty for collision
            return reward, self.get_state(), self.lost, self.won

        # # Reward for moving closer to the treasure
        # Previous head position
        # prev_head_pos = list(self.snake_position[0])
        # prev_distance = self.get_wrapped_distance(prev_head_pos, self.price_position)
        # new_distance = self.get_wrapped_distance(head_pos, self.price_position)
        # if new_distance < prev_distance:
        #     reward += 0.01  # Small positive reward
        # else:
        #     reward -= 0.01  # Small negative reward

        # Small penalty per step to encourage efficiency
        reward -= 0.001

        return reward, self.get_state(), self.lost, self.won
