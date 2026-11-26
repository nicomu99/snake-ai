"""Connects a ``Game`` to the Tk ``View``."""

import logging

from snake_ai.gui.view import View
from snake_ai.session import Game

logger = logging.getLogger(__name__)


class Controller:
    """Tk adapter around a Game.

    Maps keyboard input to moves, paces the agent with Tk's ``after()`` and
    connects the view to the game's state changes.

    The game renders through the view on every move and reset, because the
    controller subscribes ``view.render`` to it.

    Args:
        game: The game to play and display.
        view: The window that renders the game and collects input.

    Attributes:
        KEY_DIRECTIONS (dict[str, str]): Maps Tk key symbols (WASD) to move
            directions.
        AGENT_STEPS (int): The maximum number of moves the agent makes each
            time the start button is pressed.
        AGENT_DELAY_MS (int): The delay between two agent moves, in
            milliseconds.
        game: The game being played.
        view: The window the game is drawn in.
    """

    KEY_DIRECTIONS = {"w": "UP", "s": "DOWN", "a": "LEFT", "d": "RIGHT"}
    AGENT_STEPS = 5
    AGENT_DELAY_MS = 100

    def __init__(self, game: Game, view: View):
        self.game = game
        self.view = view

        self.game.subscribe(view.render)
        view.on_key(self.handle_key)
        view.on_start(self.run_agent)

    def run(self):
        """Draws the current game and enters the Tk main loop.

        This call blocks until the window is closed.
        """
        self.view.render(self.game.snapshot())
        self.view.run()

    def handle_key(self, event):
        """Performs a move for a WASD key press and ignores all other keys.

        Args:
            event: The Tk key event. Its ``keysym`` selects the direction.
        """
        direction = self.KEY_DIRECTIONS.get(event.keysym)
        if direction is not None:
            self.game.step(direction)

    def run_agent(self):
        """Lets the agent play up to ``AGENT_STEPS`` moves.

        The moves are scheduled on the Tk event loop ``AGENT_DELAY_MS``
        apart, so this returns immediately and the window stays responsive.
        """
        logger.info("Running agent")
        self._agent_step(self.AGENT_STEPS)

    def _agent_step(self, steps_left):
        """Performs one agent move and schedules the next one.

        No further move is scheduled once the game is lost or won, or when
        this was the last remaining step.

        Args:
            steps_left: The number of moves left, including this one.
        """
        _reward, _state, done, won = self.game.agent_step()
        if not done and not won and steps_left > 1:
            self.view.schedule(self.AGENT_DELAY_MS, self._agent_step, steps_left - 1)
