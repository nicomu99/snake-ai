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

- [ ] 🔴 Make the policy network work on any board size. Right now `Linear(192, …)` in `ai/policy_model.py` only fits a 5×5 board (`64 channels × 3 × 1` after the two unpadded convs). One approach:
  - Use `padding="same", padding_mode="circular"` in the conv layers. Because the board wraps, this matches the game rules, and it keeps the spatial size unchanged.
  - Add a global pooling layer (e.g. `nn.AdaptiveMaxPool2d(1)`) before the `Linear` layers, so their input size no longer depends on the board.
  - On a torus only relative positions matter, so pooling loses no important information, as long as the convs' receptive field covers the board. Bigger boards need more layers or dilated convs.
  - The existing `.pt` files won't load into the new architecture, so this needs retraining (which depends on the training fixes above). A model trained on one size might also need fine-tuning before it plays well on another.
- [ ] 🟡 `sample_experience` indexes into a `deque`, which is O(n) per access for items in the middle. A list-based ring buffer or a preallocated tensor buffer would make sampling faster.

### Game & environment

- [ ] 🟡 Support seeding: `SnakeEnv` uses the global `random` module. Give each env its own `random.Random(seed)` so the `reset{seed}` command can work and games can be reproduced.
- [ ] 🟡 Finish persistence: `Game.from_dict` is an unimplemented instance method (it should be a `classmethod`), and `to_dict` leaves out `direction`. Without it a restored game can't enforce the "no reversing" rule.

### WebSocket server

- [ ] 🟢 Watch the runner's loop task: if `agent_step` raises, the task dies silently and the game just freezes. The exception only shows up when the task is awaited on shutdown. Add a done-callback that logs it (and maybe notifies the client).
- [x] 🟢 The end message's `reason` is `"collision"` for every loss, including reversing direction. Report the actual reason.
- [x] 🟡 Give every server message a `type` field. Snapshots, the init message and the end message have none (only errors do), so clients have to guess the message kind from its keys. This changes the protocol, so clients need updating too.
- [ ] 🟡 Implement the `set_model` command (currently a stub). This needs a way to list and load the models in `policy_models/`, and an agent per model instead of one shared agent.
- [ ] 🟡 `agent_step` runs the PyTorch inference synchronously on the event loop. With many games this blocks the loop and makes the ticks uneven. `asyncio.to_thread` is a medium fix. Batching inference across games is larger (🔴).
- [ ] 🟡 Limit how many games can exist. Every connection without a known `game_id` creates a runner, and anyone who knows a `game_id` can take over that game. Consider a cap on runners and/or a secret token per game.
- [ ] 🔴 Implement the Redis-backed registry: `save` and `delete` are stubs and the Redis code is commented out. Depends on `Game.from_dict` (see above).

### Local GUI

- [x] 🟢 Fix `local_main.py`: it calls `Game(SnakeEnv(), Agent())`, but `Game` now only takes `dqn_trainer.py`.
- [x] 🟢 Fix `gui/controller.py`: it expects `step`/`agent_step` to return 3 values, but they now return 4 (`reward, state, lost, won`).

### Project & tooling

- [x] 🟢 Fix the `snake-ai` script in `pyproject.toml`: it points at `snake_ai.main:main`, which only builds the FastAPI app and doesn't start a server.
- [x] 🟢 Remove the need to run from the repo root: the model is loaded from the relative path `policy_models/model.pt` and logs go to `./app.log`, so running from another directory fails to find the model and writes logs in the wrong place.
- [x] 🟢 Make the logging setup configurable: the root level is hard-coded to `DEBUG` and the log file path to `app.log`. Read both from settings or environment variables.
- [ ] 🟡 Add tests. There are none yet. Good first targets are `SnakeEnv` (moves, wrapping, eating, winning, losing), `parse_message`, and the `GameRunner` lifecycle (attach, detach, expire).
