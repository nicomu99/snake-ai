from model import Model
from view import View
from utils import Utils
from tkinter import *
from threading import Thread
from ai.agent import Agent


class Controller:
    def __init__(self):
        print("Initializing Controller")
        self.model = Model()
        self.show_gui = False
        self.view = None
        self.agent = Agent()
        self.agent.train_or_load_model(self)
        self.setup_gui()

    def reset(self):
        return self.model.reset()

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

        # create a button
        button = Button(root, text="Start", command=self.run_agent)
        button.grid(column=0, row=1, sticky='N W E S')

    def run_agent(self):
        print("Running Agent")
        Thread(target=self.agent.play_steps, args=(self,)).start()

    def run_gui(self):
        # run the gui
        self.show_gui = True
        self.view.mainloop()

    def perform_move(self, direction):
        # refactor the direction to "UP", "DOWN", "LEFT", "RIGHT"
        if type(direction) is not str:
            if direction == 0:
                direction = "UP"
            elif direction == 1:
                direction = "DOWN"
            elif direction == 2:
                direction = "LEFT"
            elif direction == 3:
                direction = "RIGHT"

        reward, state, done = self.model.perform_move(direction)
        if self.show_gui:
            if not self.model.lost:
                self.view.draw_snake(self.model.snake_position)
                self.view.draw_price(self.model.price_position)
            else:
                self.view.show_game_over_message(self.model.score)
        return reward, state, done


