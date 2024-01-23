from numpy.random import randint
from utils import Utils
import copy


class Model:
    def __init__(self):
        self.snake_position = [[0, 0]]
        self.last_snake_position = self.snake_position[0]
        self.direction = None
        self.lost = False
        self.score = 0

        self.price_position = [
            randint(1, Utils.TILE_COUNT.value),
            randint(1, Utils.TILE_COUNT.value)
        ]

    def is_price_collision(self):
        return (self.snake_position[0][0] == self.price_position[0] and
                self.snake_position[0][1] == self.price_position[1])

    def is_snake_collision(self):
        return self.snake_position[0] in self.snake_position[1:]

    def game_over(self):
        self.lost = True

    def handle_collision(self):
        # generate new random location of price
        new_pricex = randint(0, Utils.TILE_COUNT.value)
        new_pricey = randint(0, Utils.TILE_COUNT.value)
        while [new_pricex, new_pricey] in self.snake_position:
            new_pricex = randint(0, Utils.TILE_COUNT.value)
            new_pricey = randint(0, Utils.TILE_COUNT.value)
        self.price_position = [new_pricex, new_pricey]

        # add new square to snake
        self.snake_position.append(self.last_snake_position)

    def perform_move(self, direction):
        # update the current direction
        if (self.direction == "DOWN" and direction == "UP" or
                self.direction == "RIGHT" and direction == "LEFT" or
                self.direction == "UP" and direction == "DOWN" or
                self.direction == "LEFT" and direction == "RIGHT"):
            return self.lost

        self.last_snake_position = copy.copy(self.snake_position[-1])

        # update the snake position
        snake_pos_copy = copy.deepcopy(self.snake_position)
        for i in range(1, len(self.snake_position)):
            self.snake_position[i] = snake_pos_copy[i - 1]

        # update snake position
        self.direction = direction
        head_pos = self.snake_position[0]
        if direction == "DOWN":
            head_pos[1] = (head_pos[1] + 1) % Utils.TILE_COUNT.value
        elif direction == "UP":
            head_pos[1] = (head_pos[1] - 1) % Utils.TILE_COUNT.value
        elif direction == "LEFT":
            head_pos[0] = (head_pos[0] - 1) % Utils.TILE_COUNT.value
        elif direction == "RIGHT":
            head_pos[0] = (head_pos[0] + 1) % Utils.TILE_COUNT.value
        else:
            raise Exception(f"Invalid direction detected {direction}")

        # check collision with price
        if self.is_price_collision():
            self.handle_collision()
            self.score += 1
            return self.lost

        # check collision with snake
        if self.is_snake_collision():
            self.game_over()
            return self.lost

        return self.lost
