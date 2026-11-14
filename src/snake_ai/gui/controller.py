import logging

logger = logging.getLogger(__name__)


class Controller:
    """Tk adapter around a GameSession.

    Maps keyboard input to moves, paces the agent with Tk's ``after()`` and
    connects the view to the session's state changes.
    """

    KEY_DIRECTIONS = {"w": "UP", "s": "DOWN", "a": "LEFT", "d": "RIGHT"}
    AGENT_STEPS = 5
    AGENT_DELAY_MS = 100

    def __init__(self, session, view):
        self.session = session
        self.view = view

        session.subscribe(view.render)
        view.on_key(self.handle_key)
        view.on_start(self.run_agent)

    def run(self):
        self.view.render(self.session.snapshot())
        self.view.run()

    def handle_key(self, event):
        direction = self.KEY_DIRECTIONS.get(event.keysym)
        if direction is not None:
            self.session.step(direction)

    def run_agent(self):
        logger.info("Running agent")
        self._agent_step(self.AGENT_STEPS)

    def _agent_step(self, steps_left):
        _reward, _state, done, won = self.session.agent_step()
        if not done and not won and steps_left > 1:
            self.view.schedule(self.AGENT_DELAY_MS, self._agent_step, steps_left - 1)
