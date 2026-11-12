## TODO

- [ ] Fix `local_main.py`: it calls `Game(SnakeEnv(), Agent())`, but `Game` now only takes `agent`.
- [ ] Fix training and the GUI controller: `Agent.play_one_step`, `Agent.train_or_load_model` and `gui/controller.py` expect `step`/`agent_step` to return 3 values, but the game now returns 4 (`reward, state, lost, won`). Retraining fails until this is fixed.
- [ ] Fix the `snake-ai` script in `pyproject.toml`: it points at `snake_ai.main:main`, which only builds the FastAPI app and doesn't start a server.
- [ ] Remove the need to run from the repo root: the model is loaded from the relative path `policy_models/model.pt` and logs go to `./app.log`, so running from another directory fails to find the model and writes logs in the wrong place.
