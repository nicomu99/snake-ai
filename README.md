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

- [x] 🟡 Support seeding: `SnakeEnv` uses the global `random` module. Give each env its own `random.Random(seed)` so the `reset{seed}` command can work and games can be reproduced.
- [x] 🟡 Finish persistence: `Game.from_dict` is an unimplemented instance method (it should be a `classmethod`), and `to_dict` leaves out `direction`. Without it a restored game can't enforce the "no reversing" rule.

### WebSocket server

- [x] 🟢 Watch the runner's loop task: if `agent_step` raises, the task dies silently and the game just freezes. The exception only shows up when the task is awaited on shutdown. Add a done-callback that logs it (and maybe notifies the client).
- [ ] 🟡 Implement the `set_model` command (currently a stub). This needs a way to list and load the models in `policy_models/`, and an agent per model instead of one shared agent.
- [ ] 🟢 Tell the client when `set_model` pauses the game. `GameRunner.set_model` pauses silently, so the client still shows the game as running. Send a confirmation message (e.g. `{"type": "model_changed", "model_id": …}` that also says the game is paused), and decide whether switching models should reset the game.
- [x] 🟢 Make the default model configurable and check it at startup. `PolicySelector.choose_default` hard-codes `"dqn-200k"`. If that model's file is missing, `from_path` only skips it with a warning, so the server starts but every new connection fails in `GameRunner.__init__`. Move the default into `models.json` (e.g. a top-level `"default"` field) and make `from_path` fail if the default model can't be loaded.
- [ ] 🟡 `agent_step` runs the PyTorch inference synchronously on the event loop. With many games this blocks the loop and makes the ticks uneven. `asyncio.to_thread` is a medium fix. Batching inference across games is larger (🔴).
- [ ] 🟡 Limit how many games can exist. Every connection without a known `game_id` creates a runner, and anyone who knows a `game_id` can take over that game. Consider a cap on runners and/or a secret token per game.
- [ ] 🔴 Implement the Redis-backed registry: `save` and `delete` are stubs and the Redis code is commented out. Depends on `Game.from_dict` (see above).

### Testing

- [ ] 🟡 Add tests. There are none yet. Good first targets are `SnakeEnv` (moves, wrapping, eating, winning, losing), `parse_message`, and the `GameRunner` lifecycle (attach, detach, expire).
