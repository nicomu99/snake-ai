from tkinter import *
# from tkinter import ttk
import numpy as np
from utils import Utils
from tkinter import messagebox


class View(Canvas):
    def __init__(self, parent, snake_position, price_position, controller, **kwargs):
        super().__init__(parent, **kwargs)
        parent.bind("<KeyPress>", self.move)

        # Create Background
        for i in range(Utils.TILE_COUNT.value):
            for j in range(Utils.TILE_COUNT.value):
                self.create_rectangle(
                    (
                        self.calculate_position(i),
                        self.calculate_position(j),
                        self.calculate_position(i + 1),
                        self.calculate_position(j + 1)
                    ),
                    fill="white"
                )

        self.controller = controller

        snake_posx, snake_posy = snake_position[0]
        self.snake_rect_ids = [self.create_rectangle(
            self.calculate_position(snake_posx),
            self.calculate_position(snake_posy),
            self.calculate_position(snake_posx + 1),
            self.calculate_position(snake_posy + 1),
            fill="red"
        )]

        price_posx, price_posy = price_position
        self.price_rect_id = self.create_rectangle(
            self.calculate_position(price_posx),
            self.calculate_position(price_posy),
            self.calculate_position(price_posx + 1),
            self.calculate_position(price_posy + 1),
            fill="yellow"
        )

    def draw_snake(self, snake_position):
        # check if we have to add a new snake rect
        if len(snake_position) > len(self.snake_rect_ids):
            posx, posy = snake_position[-1]
            self.snake_rect_ids.append(
                self.create_rectangle(
                    self.calculate_position(posx),
                    self.calculate_position(posy),
                    self.calculate_position(posx + 1),
                    self.calculate_position(posy + 1),
                    fill="red"
                )
            )
        for [posx, posy], rect_id in zip(snake_position, self.snake_rect_ids):
            self.coords(
                rect_id,
                self.calculate_position(posx),
                self.calculate_position(posy),
                self.calculate_position(posx + 1),
                self.calculate_position(posy + 1)
            )

    def draw_price(self, price_position):
        posx, posy = price_position
        self.coords(
            self.price_rect_id,
            self.calculate_position(posx),
            self.calculate_position(posy),
            self.calculate_position(posx + 1),
            self.calculate_position(posy + 1)
        )

    def show_game_over_message(self, score):
        self.show_popup(score)

    @staticmethod
    def show_popup(score):
        messagebox.showinfo("Game lost!", f"You have lost the game! Your score: {score}")

    @staticmethod
    def calculate_position(tile_id):
        return tile_id * Utils.TILE_SIZE.value

    def move(self, event):
        if event.keysym == "w":
            self.controller.perform_move("UP")
        elif event.keysym == "s":
            self.controller.perform_move("DOWN")
        elif event.keysym == "a":
            self.controller.perform_move("LEFT")
        elif event.keysym == "d":
            self.controller.perform_move("RIGHT")
