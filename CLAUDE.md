# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Snake game played by a Double-DQN agent (PyTorch), served to clients over a FastAPI WebSocket. There is also a legacy local Tkinter GUI. Managed with `uv` (`src/` layout, package `snake_ai`). There are no tests yet.

## Commands

```bash
uv sync                                              # install deps (incl. dev group: ruff)
uv run uvicorn snake_ai.main:main_app --reload       # run the WebSocket server
uv run ruff check src                                # lint (rules E, F, I, B, W; line length 88)
uv run ruff check --fix src                          # lint + autofix (incl. import sorting)
uv run python -m snake_ai.local_main                 # local Tk GUI, plays with policy_models/model.pt
uv run python -m snake_ai.local_main --model FILE    # play with another model (name in the model dir, or absolute path)
uv run python -m snake_ai.local_main --train         # train a new policy, save it, then play
uv run python -m snake_ai.local_main --seed N        # seed torch, food placement and training (default 42)
```

- There is no `[project.scripts]` entry; run the commands above. The ASGI app object is `snake_ai.main:main_app` (`main()` only *builds* the app).
- Settings come from `SNAKE_AI_*` environment variables (`utils/settings.py`, pydantic-settings, read via the cached `get_settings()`): `SNAKE_AI_LOG_LEVEL` (default `INFO`, must be upper case), `SNAKE_AI_LOG_PATH` (default `<repo>/logs/app.log`), `SNAKE_AI_MODEL_PATH` (model directory, default `<repo>/policy_models`). Defaults are anchored to the repo root via `PROJECT_ROOT` (derived from `__file__`, so only valid for a checkout / editable install); relative env values are resolved against the cwd. Commands therefore work from any directory.

## Architecture

**Layers:** `snake/` (game rules) → `ai/` (policy + trainer) → `api/` (server sessions), with `gui/` as an alternative frontend. `utils/constants.py` holds the board size (`TILE_COUNT`, currently 5×5) and the GUI tile size; `utils/` also has the settings and the logging setup (`configure_logging()`, called once by each entry point).

- `snake/snake_env.py` — `SnakeEnv`: raw game state and rules. The board wraps at the edges (it's a torus). Reversing direction counts as a loss. `perform_move` returns `(reward, state, lost, won)` and also applies reward shaping (reversing −10, distance-to-food ±1, a −0.1 step cost, food +20, self-collision −20, win +100). State is `(snake_position, price_position, direction)`; `direction` is `None` before the first move. "price" means food/treasure throughout the code.
- `snake/game.py` — `Game(policy)`: a headless wrapper around `SnakeEnv` + a `Policy`. Maps action ints to directions via `ACTIONS = ("UP","DOWN","LEFT","RIGHT")`, counts steps, produces a serializable `snapshot()` dict, and notifies subscribed listeners on each step. Its `reset()`/`step()` make it usable as the training environment.
- `ai/policy.py` — `Policy`: wraps the network. `encode_state` turns the plain state into a `(N, N, 7)` tensor (head / body / food / four one-hot direction channels). `act()` is the greedy inference entry point. `Policy.load_model(model_file, model_path)` loads a saved state dict.
- `ai/dqn_trainer.py` — `DQNTrainer(policy)`: Double DQN with a target network, experience replay and epsilon-greedy exploration. Only needed for training; `train(env, model_path)` saves `<timestamp>_model_<steps>.pt` into `model_path`.
- `ai/` takes paths as arguments and does not read settings; the entry points (`local_main.py`, `main.py`) call `get_settings()` and pass paths down.
- `ai/policy_model.py` — a CNN whose `Linear(192, …)` input size is hard-coded for a 5×5 board. **Changing `TILE_COUNT` or the number of input channels breaks the model and makes the saved `.pt` files incompatible.**

**Server (`api/`)** — the `lifespan` in `main.py` loads `model.pt` from the model directory as one `Policy` (inference only, no `DQNTrainer`), stores it on `app.state.policy` and shares it with every game via `GameRunnerRegistry` → `GameRunner` → `Game`:
- `sockets.py` — the single endpoint `WS /ws/play?game_id=…`. Incoming JSON is parsed by `command_message.parse_message` into a pydantic discriminated union on `type`: `start`, `pause`, `resume`, `reset{seed?}`, `speed{tick_ms 1..1000}`, `set_model{model_id}`. Invalid input gets `{"type":"error","message":…}` back.
- `GameRunnerRegistry` — maps ids to `GameRunner`s (keys are `game_runner:<id>`). Connecting without a `game_id`, or with an unknown one, creates a new runner. Connecting with a known id re-attaches to that runner. The commented-out Redis code shows a planned persistence backend.
- `GameRunner` — owns one `Game` and an asyncio loop task. The loop waits on the `running` Event, then each tick calls `agent_step`, sends the snapshot to the attached `WebSocketSink`, and sleeps for `tick_ms`. It pauses and sends an end message when the game is lost or won. Only one sink can be attached at a time: a new attach closes the old socket with code 4000 ("opened in another tab"). The first frame sent on attach/reset is the snapshot plus `game_id` (`init_message()`).
- `GameRunnerSweeper` — a background task that, every 5 s, cancels and removes runners that have been detached longer than `GameRunner._DELETION_DELAY`.

## Known incomplete / broken areas

- If `agent_step` raises inside a `GameRunner` loop task, the task dies silently and the game freezes; the exception only surfaces on shutdown.
- `set_model`, `Game.from_dict` and the Redis-backed registry methods (`save`/`delete`) are stubs/TODOs.
