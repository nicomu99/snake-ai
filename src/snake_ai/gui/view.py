from tkinter import Button, Canvas, Tk, font, messagebox

from snake_ai.utils import TileMap


class View:
    """Tk window: renders session snapshots and forwards user input."""

    def __init__(self):
        self.root = Tk()
        self.root.tk.call("tk", "scaling", 163 / 72)
        for name in (
            "TkDefaultFont",
            "TkTextFont",
            "TkFixedFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkIconFont",
            "TkTooltipFont",
        ):
            font.nametofont(name).configure(size=24)

        self.root.title("Snake")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = Canvas(self.root)
        self.canvas.grid(column=0, row=0, sticky="N W E S")

        self.root.geometry(
            f"{TileMap.TILE_COUNT.value * TileMap.TILE_SIZE.value * 10}x"
            f"{TileMap.TILE_COUNT.value * TileMap.TILE_SIZE.value * 10}")

        self.start_button = Button(self.root, text="Start")
        self.start_button.grid(column=0, row=1, sticky="N W E S")

        # Create Background
        for i in range(TileMap.TILE_COUNT.value):
            for j in range(TileMap.TILE_COUNT.value):
                self.create_tile(i, j, "white")

        # Positioned by the first render()
        self.snake_rect_ids = []
        self.price_rect_id = self.create_tile(0, 0, "yellow")

    def on_key(self, handler):
        self.root.bind("<KeyPress>", handler)

    def on_start(self, handler):
        self.start_button.configure(command=handler)

    def schedule(self, delay_ms, func, *args):
        return self.root.after(delay_ms, func, *args)

    def run(self):
        self.root.mainloop()

    def render(self, snapshot):
        if snapshot["lost"]:
            self.show_game_over_message(snapshot["score"])
            return
        self.draw_snake(snapshot["snake"])
        self.draw_price(snapshot["food"])

    def draw_snake(self, snake_position):
        # Keep exactly one rect per snake segment
        while len(self.snake_rect_ids) < len(snake_position):
            self.snake_rect_ids.append(self.create_tile(0, 0, "red"))
        while len(self.snake_rect_ids) > len(snake_position):
            self.canvas.delete(self.snake_rect_ids.pop())

        for (posx, posy), rect_id in zip(
                snake_position, self.snake_rect_ids, strict=True
        ):
            self.move_tile(rect_id, posx, posy)

    def draw_price(self, price_position):
        posx, posy = price_position
        self.move_tile(self.price_rect_id, posx, posy)

    def show_game_over_message(self, score):
        messagebox.showinfo(
            "Game lost!", f"You have lost the game! Your score: {score}")

    def create_tile(self, posx, posy, fill):
        return self.canvas.create_rectangle(*self.tile_bounds(posx, posy), fill=fill)

    def move_tile(self, rect_id, posx, posy):
        self.canvas.coords(rect_id, *self.tile_bounds(posx, posy))

    @classmethod
    def tile_bounds(cls, posx, posy):
        return (
            cls.calculate_position(posx), cls.calculate_position(posy),
            cls.calculate_position(posx + 1), cls.calculate_position(posy + 1),
        )

    @staticmethod
    def calculate_position(tile_id):
        return tile_id * TileMap.TILE_SIZE.value * 10
