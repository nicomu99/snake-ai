# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Snake game in Tkinter, plus a Double DQN agent in TensorFlow/Keras that learns to play it. There are no tests, linter or CI config yet.

## Environment and running

- Python 3.12. The project is moving to `uv` (`pyproject.toml`, `uv.lock`, `.venv/`). The old `venv/` directory is legacy.
- **The `uv` setup is incomplete.** `src/snake_ai/__init__.py` is only the `uv init` placeholder (`main()` prints "Hello from snake-ai!"), so the `snake-ai` script entry point does not run the game. The real code is still in the top-level `src/snake_ai/snake` and `src/snake_ai/ai` packages.
- `tensorflow` is imported by `src/snake_ai/ai` but is **not** declared in `pyproject.toml` or installed in `.venv`. Add it with `uv add tensorflow` before running.
- The imports are mixed: `src/snake_ai/snake` and `src/snake_ai/ai` use package imports (`from snake.controller ...`, `from snake.utils ...`), but `src/snake_ai/snake`, `model.py` and `view.py` use bare imports (`from model import Model`). So both the repo root and `src/snake_ai/snake` must be on the path:

  ```sh
  PYTHONPATH=.:snake uv run python -m snake.main
  ```

- Run from the repo root. The agent loads and saves `model.keras` through a relative path.

## Architecture

MVC-style game, with the RL agent driving the game through the controller:

- `src/snake_ai/snake` (`Model`) holds the game state and the **RL environment logic**. `perform_move(direction)` returns `(reward, state, done)`, and the reward shaping lives here: +20 for food, -20 for self-collision, ±1 for moving closer to or farther from the food (wrapped Euclidean distance), -0.1 per step, and -10 for trying to reverse direction (the move is then ignored). `get_state()` returns a `(TILE_COUNT, TILE_COUNT, 3)` float grid with channels for head, body and food. The board **wraps around** at the edges, so walls never kill the snake.
- `src/snake_ai/snake` (`View`, a Tk `Canvas`) only renders rectangles.
- `src/snake_ai/snake` (`Controller`) is the hub. Its constructor creates the `Model` and the `Agent`, then calls `agent.train_or_load_model(self)`, which **blocks: it trains synchronously before the GUI exists** whenever `model.keras` is missing. After that it builds the Tk window. `perform_move` accepts either a direction string (from the WASD keys) or an action int 0–3 (UP, DOWN, LEFT, RIGHT) from the agent. It redraws only when `show_gui` is true, which `run_gui()` sets, so training runs headless.
- `src/snake_ai/ai` (`Agent`) is a Double DQN with a Conv2D model, a replay buffer and a target network. It uses the controller as its environment interface (`controller.reset()`, `controller.perform_move(action)`). The "Start" button runs `play_steps` on a background thread, which plays only 5 greedy moves.
- `src/snake_ai/snake`: the `Utils` enum holds `TILE_COUNT` (board size, currently 5) and `TILE_SIZE` (pixels). Changing `TILE_COUNT` changes the network's input shape, so the saved `model.keras` becomes incompatible. Delete it to retrain.

Known quirks:
- Food placement uses `randint(1, TILE_COUNT)` on reset but `randint(0, TILE_COUNT)` after eating. numpy's upper bound is exclusive, so the ranges differ.
