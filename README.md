# Snake AI

# Setup

Synchronize dependencies: 

```bash
uv sync
```

Run websocket

```bash
uv run uvicorn snake_ai.main:main_app --reload
```

Run local GUI

```bash
uv run python -m snake_ai.local_main
```

Run test client

```bash
uv run python3 -m websockets ws://127.0.0.1:8000/ws/play
```


## TODO

Effort: 🟢 quick (under an hour) · 🟡 medium (a few hours) · 🔴 large (a day or more, or needs retraining)

Within each category, bugs come first, then unfinished features, then cleanup.

### Agent & training

- [x] 🟢 Fix training for the current `Game` API: `Agent.play_one_step` and `Agent.train_or_load_model` expect `step` to return 3 values, but it now returns 4 (`reward, state, lost, won`). Retraining fails until this is fixed.
- [x] 🟢 Fix epsilon decay in `Agent.train`: epsilon is multiplied by `0.99` on every *step* (the comment says "after each episode"), so it hits `epsilon_min` after about 230 steps. That's long before `learning_starts` (10,000), so the agent barely explores. Decay per episode, or decay linearly over a large share of `train_steps`.
- [x] 🟢 Fix the duplicate training update in `Agent.train`: `train_from_experience()` also runs on every step once the buffer holds `batch_size` items. That skips the `learning_starts` / `train_policy_steps` gating and trains up to twice per step. Remove the second call.
- [x] 🟢 Fix the score logging in `Agent.train`: it divides by `len(avg_score)`, which raises `ZeroDivisionError` if no episode has ended yet. `avg_score` is also never cleared, so it logs an all-time average instead of a recent one.
- [x] 🟢 Don't fail silently in `Agent.load_model`: every exception is swallowed, so the server starts with an untrained random agent and only prints "No Model Found". Log the error and fail startup instead. Also pass `map_location=self.device` to `torch.load`, so a model saved on a GPU loads on a CPU-only machine.
- [ ] 🟡 The agent can't see the snake's direction. For a snake of length 1 there is no body to show it, but reversing still counts as a loss, so the agent can lose a game it had no way to avoid. Changing the rule so reversing isn't a loss at length 1 is quick. Adding a direction channel to `encode_state` changes the model input, so it needs retraining (🔴).
- [ ] 🔴 Make the policy network work on any board size. Right now `Linear(192, …)` in `ai/policy_model.py` only fits a 5×5 board (`64 channels × 3 × 1` after the two unpadded convs). One approach:
  - Use `padding="same", padding_mode="circular"` in the conv layers. Because the board wraps, this matches the game rules, and it keeps the spatial size unchanged.
  - Add a global pooling layer (e.g. `nn.AdaptiveMaxPool2d(1)`) before the `Linear` layers, so their input size no longer depends on the board.
  - On a torus only relative positions matter, so pooling loses no important information, as long as the convs' receptive field covers the board. Bigger boards need more layers or dilated convs.
  - The existing `.pt` files won't load into the new architecture, so this needs retraining (which depends on the training fixes above). A model trained on one size might also need fine-tuning before it plays well on another.
- [ ] 🟡 `sample_experience` indexes into a `deque`, which is O(n) per access for items in the middle. A list-based ring buffer or a preallocated tensor buffer would make sampling faster.
- [x] 🟢 Deduplicate the loading code in `Agent.train_or_load_model` (it repeats `load_model`), make the model path configurable, and create `policy_models/` before saving.
- [x] 🟢 Clean up the agent: replace the `print` calls with `logger`, remove the unused `input_space`, and call `self.sequential(x)` directly in `PolicyModel.forward` instead of looping over the layers.

### Game & environment

- [x] 🟢 Fix the win return in `SnakeEnv.perform_move`: it returns `self.get_state` (the method) instead of `self.get_state()`, so callers get a function instead of a state when the snake wins. `encode_state` would crash on it during training.
- [x] 🟢 Fix the final score: when the last food wins the game, `perform_move` returns before `self.score += 1`, so the winning score is one too low.
- [x] 🟢 Block moves after the game ends: neither `Game.step` nor `SnakeEnv.perform_move` checks `lost`/`won`, so the snake keeps moving and scoring after a game over.
- [x] 🟢 Validate the direction before changing state: `perform_move` shifts the body before it raises on an unknown direction, which leaves the snake broken. In `Game.step`, an out-of-range int raises `IndexError`, and a negative int silently picks a direction (`ACTIONS[-1]` is `"RIGHT"`).
- [x] 🟢 Food can spawn on the snake's head at `[0, 0]` on init/reset. It isn't eaten until the snake leaves the tile and comes back. Pick the first food position from the free tiles, like `handle_collision` does.
- [x] 🟢 `Game.reset` doesn't notify listeners, so a subscribed view (e.g. the Tk GUI) shows the old board until the next move.
- [x] 🟢 `SnakeEnv.get_state` returns the env's own lists, which later moves mutate. Anyone who keeps a state around (e.g. for replay or history) needs a copy.
- [ ] 🟡 Support seeding: `SnakeEnv` uses the global `random` module. Give each env its own `random.Random(seed)` so the `reset{seed}` command can work and games can be reproduced.
- [ ] 🟡 Finish persistence: `Game.from_dict` is an unimplemented instance method (it should be a `classmethod`), and `to_dict` leaves out `direction`. Without it a restored game can't enforce the "no reversing" rule.
- [x] 🟢 `TileMap` is an `Enum`, so members with equal values become aliases. For example, if `TILE_COUNT` and `TILE_SIZE` were both 30, `TileMap.TILE_SIZE` would silently become `TileMap.TILE_COUNT`. Use plain constants or a frozen dataclass instead.
- [x] 🟢 Clean up `SnakeEnv`: `reward_decay`, `min_reward` and `prev_direction` are never read (and `prev_direction` grows without limit), `__init__` repeats `reset`, and the commented-out print at the end of `perform_move` can go. Move the reward values into named constants.

### WebSocket server

- [x] 🟢 Fix `GameRunnerRegistry.close_all`: `zip(runners, results)` yields `((id, runner), result)` pairs, but the loop unpacks three values. That raises `ValueError` on shutdown whenever at least one game exists. Use `for (runner_id, _), result in zip(...)`.
- [x] 🟢 Always detach in `sockets.play`: only `WebSocketDisconnect` leads to `registry.detach`. Any other exception (e.g. a binary frame, which makes `receive_json` raise `KeyError`, or an error inside `session.handle`) ends the handler without detaching. The runner then never expires and keeps a dead sink. Use `try/finally`.
- [ ] 🟢 Watch the runner's loop task: if `agent_step` raises, the task dies silently and the game just freezes. The exception only shows up when the task is awaited on shutdown. Add a done-callback that logs it (and maybe notifies the client).
- [x] 🟢 `GameRunner.reset` calls `self.sink.send_frame` directly, which raises `AttributeError` when no sink is attached. Use `self.send_frame`. It also only sends `{"game_id"}`, so the client doesn't see the new board until the game is started. Send a snapshot too.
- [x] 🟢 Fix behavior when calling start when the game is already finished. Right now the loop makes another move on the finished game, sends the end message again and pauses. Depends on "Block moves after the game ends".
- [x] 🟢 `Start` and `Resume` do the same thing. Merge them, or give `start` its own meaning (e.g. reset the game if it is finished).
- [ ] 🟢 The end message's `reason` is `"collision"` for every loss, including reversing direction. Report the actual reason.
- [x] 🟢 Handle the `RuntimeError` in `GameRunner.attach` (the `# TODO` when closing the old sink) instead of ignoring it, or at least log it.
- [ ] 🟡 Give every server message a `type` field. Snapshots, the init message and the end message have none (only errors do), so clients have to guess the message kind from its keys. This changes the protocol, so clients need updating too.
- [ ] 🟡 Implement the `set_model` command (currently a stub). This needs a way to list and load the models in `policy_models/`, and an agent per model instead of one shared agent.
- [ ] 🟡 `agent_step` runs the PyTorch inference synchronously on the event loop. With many games this blocks the loop and makes the ticks uneven. `asyncio.to_thread` is a medium fix. Batching inference across games is larger (🔴).
- [ ] 🟡 Limit how many games can exist. Every connection without a known `game_id` creates a runner, and anyone who knows a `game_id` can take over that game. Consider a cap on runners and/or a secret token per game.
- [ ] 🔴 Implement the Redis-backed registry: `save` and `delete` are stubs and the Redis code is commented out. Depends on `Game.from_dict` (see above).

### Local GUI

- [ ] 🟢 Fix `local_main.py`: it calls `Game(SnakeEnv(), Agent())`, but `Game` now only takes `dqn_trainer.py`.
- [ ] 🟢 Fix `gui/controller.py`: it expects `step`/`agent_step` to return 3 values, but they now return 4 (`reward, state, lost, won`).

### Project & tooling

- [ ] 🟢 Fix the `snake-ai` script in `pyproject.toml`: it points at `snake_ai.main:main`, which only builds the FastAPI app and doesn't start a server.
- [ ] 🟢 Remove the need to run from the repo root: the model is loaded from the relative path `policy_models/model.pt` and logs go to `./app.log`, so running from another directory fails to find the model and writes logs in the wrong place.
- [ ] 🟢 Make the logging setup configurable: the root level is hard-coded to `DEBUG` and the log file path to `app.log`. Read both from settings or environment variables.
- [ ] 🟡 Add tests. There are none yet. Good first targets are `SnakeEnv` (moves, wrapping, eating, winning, losing), `parse_message`, and the `GameRunner` lifecycle (attach, detach, expire).
