from model import Model
from view import View
from utils import Utils
from tkinter import *
from threading import Thread


class Controller:
    def __init__(self):
        print("Initializing Controller")
        self.model = Model()

        # set up the view
        self.root = Tk()
        self.root.title("Snake")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.view = View(self.root, self.model.snake_position, self.model.price_position, self)
        self.view.grid(column=0, row=0, sticky='N W E S')

        self.root.geometry(f"{Utils.TILE_COUNT.value * Utils.TILE_SIZE.value}x"
                           f"{Utils.TILE_COUNT.value * Utils.TILE_SIZE.value}")

    def perform_move(self, direction):
        # refactor the direction to "UP", "DOWN", "LEFT", "RIGHT"
        if direction[0][0] == 1:
            direction = "UP"
        elif direction[0][1] == 1:
            direction = "DOWN"
        elif direction[0][2] == 1:
            direction = "LEFT"
        elif direction[0][3] == 1:
            direction = "RIGHT"

        reward, state = self.model.perform_move(direction)
        if not self.model.lost:
            self.view.draw_snake(self.model.snake_position)
            self.view.draw_price(self.model.price_position)
        else:
            self.view.show_game_over_message(self.model.score)
        return reward, state


