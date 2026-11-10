from snake_ai.ai import Agent
from snake_ai.gui import Controller, View
from snake_ai.snake import GameSession, Model
from snake_ai.utils import configure_logging


def main():
    configure_logging()

    session = GameSession(Model(), Agent())
    session.agent.train_or_load_model(session)

    Controller(session, View()).run()


if __name__ == '__main__':
    main()
