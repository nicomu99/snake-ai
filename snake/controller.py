from model import Model
from view import View
from utils import Utils
from tkinter import *


class Controller:
    def __init__(self):
        self.model = Model()

        # set up the view
        root = Tk()
        root.title("Snake")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.view = View(root, self.model.snake_position, self.model.price_position, self)
        self.view.grid(column=0, row=0, sticky='N W E S')

        root.geometry(f"{Utils.TILE_COUNT.value * Utils.TILE_SIZE.value}x"
                      f"{Utils.TILE_COUNT.value * Utils.TILE_SIZE.value}")
        root.mainloop()

    def perform_move(self, direction):
        # TODO: Implement the correct return in the model
        reward, state = self.model.perform_move(direction)
        return reward, state

        # if not self.model.lost:
        #     self.view.draw_snake(self.model.snake_position)
        #     self.view.draw_price(self.model.price_position)
        # else:
        #     self.view.show_game_over_message(self.model.score)
