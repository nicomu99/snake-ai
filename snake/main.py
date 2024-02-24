from ai.agent import Agent
from snake.controller import Controller
import tensorflow as tf

if __name__ == '__main__':
    # agent = Agent()

    controller = Controller()
    controller.run_gui()
    # for i in range(100):
    #     action = agent.get_move(controller.model.get_state())
    #     controller.perform_move(action)


