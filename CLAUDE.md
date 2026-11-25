# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Snake game played by a Double-DQN agent (PyTorch), served to clients over a FastAPI WebSocket. There is also a legacy local Tkinter GUI. Managed with `uv` (`src/` layout, package `snake_ai`). There are no tests yet.

## Commands

```bash
uv run uvicorn snake_ai.main:main_app --reload       # run the WebSocket server
uv run ruff check --fix src                          # lint + autofix (incl. import sorting)
uv run python -m snake_ai.local_main --help          # local Tk GUI: --model, --train, --seed + training hyperparameters
```

- There is no `[project.scripts]` entry. The ASGI app object is `snake_ai.main:main_app` (`main()` only *builds* the app).
- Settings come from `SNAKE_AI_*` env vars (`utils/settings.py`, read via the cached `get_settings()`). `SNAKE_AI_LOG_LEVEL` must be upper case. `SNAKE_AI_MODEL_CATALOG` is computed after the env is read, so it follows `SNAKE_AI_MODEL_PATH`. Defaults are anchored to the repo root via `PROJECT_ROOT` (derived from `__file__`, so only valid for a checkout / editable install); relative env values resolve against the cwd.

## Architecture

**Layers:** `snake/` (game rules) → `ai/` (policy + trainer) → `api/` (server sessions), with `gui/` as an alternative frontend. `ai/` takes paths as arguments and does not read settings; the entry points (`local_main.py`, `main.py`) call `get_settings()` and pass paths down. Each entry point calls `configure_logging()` once.

- Game rules: the board wraps at the edges (a torus), and reversing direction counts as a loss. `direction` is `None` before the first move. "price" means food/treasure throughout the code.
- **`ai/policy_model.py` sizes its first `Linear` layer from `TILE_COUNT` (`utils/constants.py`). Changing `TILE_COUNT`, the number of input channels or the layers makes saved `.pt` files incompatible.** `frontend/src/config.ts` `BOARD_SIZE` must match `TILE_COUNT`.
- Model catalog (`<model_path>/models.json`): missing model files are skipped with a warning, but startup fails if the `default_model` can't be loaded. Policies are shared, inference-only, one per model.
- `frontend/src/protocol.ts` mirrors the server's message types — keep it in sync with `api/`.

## Known incomplete / broken areas

- The Redis-backed registry methods (`save`/`delete`) are stubs/TODOs; the commented-out Redis code shows the planned persistence backend. `Game.to_dict`/`from_dict` (incl. the food RNG state) are ready for it.
