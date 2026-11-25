"""The Tk window that draws the Snake board."""

from tkinter import Button, Canvas, Tk, font, messagebox

from snake_ai.utils import TILE_COUNT, TILE_SIZE


class View:
    """Tk window: renders game snapshots and forwards user input.

    The window holds a canvas with a ``TILE_COUNT`` x ``TILE_COUNT`` grid of
    white tiles and a start button below it. The snake is drawn as red tiles
    and the food ("price") as a yellow tile. Creating a ``View`` also
    creates the Tk root window and sets the Tk scaling and default font
    sizes for a high-DPI screen.

    Attributes:
        root: The Tk root window.
        canvas: The canvas the board is drawn on.
        start_button: The button that starts the agent, see ``on_start``.
        snake_rect_ids: The canvas ids of the snake rectangles, one per
            segment and in the same order as the snake positions.
        price_rect_id: The canvas id of the food rectangle.
    """

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

        board_px = TILE_COUNT * TILE_SIZE * 10
        self.canvas = Canvas(
            self.root, width=board_px, height=board_px, highlightthickness=0)
        self.canvas.grid(column=0, row=0, sticky="N W E S")

        # self.root.geometry(
        #     f"{TILE_COUNT * TILE_SIZE * 10}x"
        #     f"{TILE_COUNT * TILE_SIZE * 10}")

        self.start_button = Button(self.root, text="Start")
        self.start_button.grid(column=0, row=1, sticky="N W E S")

        # Create Background
        for i in range(TILE_COUNT):
            for j in range(TILE_COUNT):
                self.create_tile(i, j, "white")

        # Positioned by the first render()
        self.snake_rect_ids = []
        self.price_rect_id = self.create_tile(0, 0, "yellow")

    def on_key(self, handler):
        """Registers a handler for key presses in the window.

        Args:
            handler: A callable that takes the Tk key event.
        """
        self.root.bind("<KeyPress>", handler)

    def on_start(self, handler):
        """Registers a handler for the start button.

        Args:
            handler: A callable without arguments.
        """
        self.start_button.configure(command=handler)

    def schedule(self, delay_ms, func, *args):
        """Calls a function later on the Tk event loop.

        Args:
            delay_ms: The delay in milliseconds.
            func: The function to call.
            *args: The positional arguments passed to ``func``.

        Returns:
            The Tk id of the scheduled call, which can be passed to
            ``root.after_cancel``.
        """
        return self.root.after(delay_ms, func, *args)

    def run(self):
        """Enters the Tk main loop. Blocks until the window is closed."""
        self.root.mainloop()

    def render(self, snapshot):
        """Draws a game snapshot.

        If the game is lost, a game-over dialog is shown instead and the
        board is left unchanged.

        Args:
            snapshot: A dict as returned by ``Game.snapshot``. Only its
                ``snake``, ``food``, ``score`` and ``lost`` keys are used.
        """
        if snapshot["lost"]:
            self.show_game_over_message(snapshot["score"])
            return
        self.draw_snake(snapshot["snake"])
        self.draw_price(snapshot["food"])

    def draw_snake(self, snake_position):
        """Moves the snake rectangles to the given positions.

        Rectangles are created or deleted so that there is exactly one per
        segment.

        Args:
            snake_position: The ``(x, y)`` tile positions of the snake
                segments.
        """
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
        """Moves the food rectangle to the given tile.

        Args:
            price_position: The ``(x, y)`` tile position of the food.
        """
        posx, posy = price_position
        self.move_tile(self.price_rect_id, posx, posy)

    def show_game_over_message(self, score):
        """Shows a modal dialog that the game is lost.

        Args:
            score: The final score shown in the dialog.
        """
        messagebox.showinfo(
            "Game lost!", f"You have lost the game! Your score: {score}")

    def create_tile(self, posx, posy, fill):
        """Draws a new tile-sized rectangle on the canvas.

        Args:
            posx: The tile column.
            posy: The tile row.
            fill: The Tk fill color.

        Returns:
            The canvas id of the new rectangle.
        """
        return self.canvas.create_rectangle(*self.tile_bounds(posx, posy), fill=fill)

    def move_tile(self, rect_id, posx, posy):
        """Moves an existing rectangle to a tile.

        Args:
            rect_id: The canvas id of the rectangle.
            posx: The tile column.
            posy: The tile row.
        """
        self.canvas.coords(rect_id, *self.tile_bounds(posx, posy))

    @classmethod
    def tile_bounds(cls, posx, posy):
        """Returns the canvas bounding box of a tile.

        Args:
            posx: The tile column.
            posy: The tile row.

        Returns:
            An ``(x0, y0, x1, y1)`` tuple in canvas pixels.
        """
        return (
            cls.calculate_position(posx), cls.calculate_position(posy),
            cls.calculate_position(posx + 1), cls.calculate_position(posy + 1),
        )

    @staticmethod
    def calculate_position(tile_id):
        """Converts a tile index into a canvas pixel coordinate.

        Each tile is ``TILE_SIZE * 10`` pixels wide and high.

        Args:
            tile_id: The tile index along one axis.

        Returns:
            The pixel coordinate of the tile's top or left edge.
        """
        return tile_id * TILE_SIZE * 10
