from model import Model
from view import View
from utils import Utils
from tkinter import *
from threading import Thread


class Controller:
    def __init__(self, show_gui=True):
        print("Initializing Controller")
        self.model = Model()
        self.show_gui = show_gui
        self.view = None

        if self.show_gui:
            self.setup_gui()

    def setup_gui(self):
        # set up the view
        root = Tk()
        root.title("Snake")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.view = View(root, self.model.snake_position, self.model.price_position, self)
        self.view.grid(column=0, row=0, sticky='N W E S')

        root.geometry(f"{Utils.TILE_COUNT.value * Utils.TILE_SIZE.value}x"
                      f"{Utils.TILE_COUNT.value * Utils.TILE_SIZE.value}")

    def run_gui(self):
        # run the gui
        self.view.mainloop()

    def perform_move(self, direction):
        # refactor the direction to "UP", "DOWN", "LEFT", "RIGHT"
        if type(direction) != str:
            if direction[0][0] == 1:
                direction = "UP"
            elif direction[0][1] == 1:
                direction = "DOWN"
            elif direction[0][2] == 1:
                direction = "LEFT"
            elif direction[0][3] == 1:
                direction = "RIGHT"

        reward, state = self.model.perform_move(direction)
        if self.show_gui:
            if not self.model.lost:
                self.view.draw_snake(self.model.snake_position)
                self.view.draw_price(self.model.price_position)
            else:
                self.view.show_game_over_message(self.model.score)
        return reward, state


