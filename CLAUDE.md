# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Snake game played by a Double-DQN agent (PyTorch), served to clients over a FastAPI WebSocket. There is also a legacy local Tkinter GUI. Managed with `uv` (`src/` layout, package `snake_ai`). There are no tests yet.

## Commands

```bash
uv sync                                              # install deps (incl. dev group: ruff)
uv run uvicorn snake_ai.main:main_app --reload       # run the WebSocket server
uv run ruff check src                                # lint (rules E, F, I, B; line length 88)
uv run ruff check --fix src                          # lint + autofix (incl. import sorting)
uv run python -m snake_ai.local_main                 # local Tk GUI (see caveats below)
```

- Run from the repo root: the model is loaded from the relative path `policy_models/model.pt`, and logs go to `./app.log`.
- The `snake-ai` script in `pyproject.toml` points at `snake_ai.main:main`, which only *builds* the FastAPI app and does not start a server. The ASGI app object is `main_app`.

## Architecture

**Layers:** `snake/` (game rules) → `ai/` (agent) → `api/` (server sessions), with `gui/` as an alternative frontend. `utils/TileMap` holds the board size (`TILE_COUNT`, currently 5×5) and tile pixel size.

- `snake/snake_env.py` — `SnakeEnv`: raw game state and rules. The board wraps at the edges (it's a torus). Reversing direction counts as a loss. `perform_move` returns `(reward, state, lost, won)` and also applies reward shaping (distance-to-food ±1, food +20, self-collision −20, win +100). State is `(snake_position, price_position)`. "price" means food/treasure throughout the code.
- `snake/game.py` — `Game`: a headless wrapper around `SnakeEnv` + `DQNTrainer`. Maps action ints to directions via `ACTIONS = ("UP","DOWN","LEFT","RIGHT")`, counts steps, produces a serializable `snapshot()` dict, and notifies subscribed listeners on each step.
- `ai/agent.py` — `DQNTrainer`: Double DQN with a policy and target network, experience replay and epsilon-greedy exploration. `encode_state` turns the plain state into a `(N, N, 3)` tensor (head / body / food channels). `act()` is the greedy inference entry point the server uses.
- `ai/policy_model.py` — a CNN whose `Linear(192, …)` input size is hard-coded for a 5×5 board. **Changing `TILE_COUNT` breaks the model and makes the saved `.pt` files incompatible.**

**Server (`api/`)** — one shared `DQNTrainer` is loaded at startup (the `lifespan` in `main.py`) and shared by every game:
- `sockets.py` — the single endpoint `WS /ws/play?game_id=…`. Incoming JSON is parsed by `command_message.parse_message` into a pydantic discriminated union on `type`: `start`, `pause`, `resume`, `reset{seed?}`, `speed{tick_ms 1..1000}`, `set_model{model_id}`. Invalid input gets `{"type":"error","message":…}` back.
- `GameRunnerRegistry` — maps ids to `GameRunner`s (keys are `game_runner:<id>`). Connecting without a `game_id`, or with an unknown one, creates a new runner. Connecting with a known id re-attaches to that runner. The commented-out Redis code shows a planned persistence backend.
- `GameRunner` — owns one `Game` and an asyncio loop task. The loop waits on the `running` Event, then each tick calls `agent_step`, sends the snapshot to the attached `WebSocketSink`, and sleeps for `tick_ms`. It pauses and sends an end message when the game is lost or won. Only one sink can be attached at a time: a new attach closes the old socket with code 4000 ("opened in another tab"). The first frame sent on attach/reset is `{"game_id": …}`.
- `GameRunnerSweeper` — a background task that, every 5 s, cancels and removes runners that have been detached longer than `GameRunner._DELETION_DELAY`.

## Known incomplete / broken areas

- The training path and the local GUI don't match the current `Game` API. `local_main.py` calls `Game(SnakeEnv(), Agent())`, but `Game.__init__` only takes `dqn_trainer.py`. `Agent.play_one_step`, `Agent.train_or_load_model` and `gui/controller.py` expect `step`/`agent_step` to return 3 values (`env.step` returns 4). Retraining needs these fixed first.
- `set_model`, the reset `seed`, `Game.from_dict` and the Redis-backed registry methods (`save`/`delete`) are stubs/TODOs.
