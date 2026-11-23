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

Run the web frontend

The frontend lives in `frontend/` (Vite + React + TypeScript) and needs Node.js with npm. Install its dependencies once:

```bash
cd frontend
npm install
```

Then start the websocket server (see above) in one terminal and the frontend dev server in a second one:

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. The dev server forwards `/ws` to the websocket server on port 8000, so both have to be running. Code changes reload automatically.

Other frontend commands (run them in `frontend/`):

```bash
npm run build    # type-check and bundle into frontend/dist/
npm run lint     # lint with oxlint
```

More details are in [frontend/README.md](frontend/README.md).

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
- [ ] 🔴 Train more agents, and ideally better ones.
  - Make the hyperparameters configurable (CLI flags or a config file) instead of hard-coding them in `DQNTrainer.__init__`, and save them next to each `.pt` file, so runs can be reproduced and compared.
  - Ideas to try: longer runs, different epsilon schedules, prioritized replay, a dueling network head, n-step returns.
  - Add each good model to `policy_models/models.json` so it shows up in the frontend's model picker.

### Deployment

- [ ] 🔴 Run the setup on a Kubernetes cluster.
  - [ ] 🟡 Dockerfiles: one for the backend (uv, with the models baked into the image or mounted from a volume), and one for the frontend (`npm run build`, served by nginx), unless FastAPI serves `frontend/dist` itself.
  - [ ] 🟡 Manifests (or a Helm chart): a Deployment and a Service for the backend, plus an Ingress that routes `/ws` to the backend with WebSocket support and `/` to the frontend. Pass the `SNAKE_AI_*` settings through a ConfigMap.
  - [ ] 🟢 Add a health endpoint for the liveness and readiness probes.
- [ ] 🔴 Implement the Redis-backed registry: `save` and `delete` are stubs and the Redis code is commented out.
  - Games live in the server process's memory, so start with one backend replica. With more replicas, reconnecting by `game_id` only works with sticky sessions or the Redis-backed registry.

### Testing

- [ ] 🟡 Add tests. There are none yet. Good first targets are `SnakeEnv` (moves, wrapping, eating, winning, losing), `parse_message`, and the `GameRunner` lifecycle (attach, detach, expire).
