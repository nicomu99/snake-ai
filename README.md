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

### Run with Docker and K8s

(optionally) Build a docker image

```bash
docker build -t <image-name>:<image-version> .
```

and push it to your docker hub.

```bash
docker push <image-name>:<image-version>
```

#### Frontend image

`frontend/Dockerfile` builds the frontend (`npm run build`) and serves `dist/` with nginx (as a non-root user, on port 8080). Like the Vite dev server, nginx forwards `/ws` and `/models` to the backend, so the app keeps talking to its own origin. Build it with `frontend/` as the build context:

```bash
docker build -t <image-name>-frontend:<image-version> frontend
```

The backend address is read from `BACKEND_URL` when the container starts (default `http://backend:8080`, i.e. a Docker network alias or a Kubernetes Service named `backend`). To try the image against a backend running locally with uvicorn on port 8000:

```bash
docker run --rm -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  <image-name>-frontend:<image-version>
```

Then open http://localhost:8080. nginx fails to start if the host in `BACKEND_URL` can't be resolved, so the backend's name must exist on the container's network. A new backend route that the frontend calls needs a `location` in `frontend/nginx/default.conf.template`, the same as in `vite.config.ts`.



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
  - [x] 🟢 Add a health endpoint for the liveness and readiness probes.
- [ ] 🔴 Implement the Redis-backed registry: `save` and `delete` are stubs and the Redis code is commented out.
  - Games live in the server process's memory, so start with one backend replica. With more replicas, reconnecting by `game_id` only works with sticky sessions or the Redis-backed registry.

### Testing

- [ ] 🟡 Add tests. There are none yet. Good first targets are `SnakeEnv` (moves, wrapping, eating, winning, losing), `parse_message`, and the `GameRunner` lifecycle (attach, detach, expire).
