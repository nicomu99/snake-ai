# Snake AI

A Snake game played by a Double-DQN agent (PyTorch) on a 5×5 board that wraps at the edges.

| Part | Where | What it does |
|---|---|---|
| Backend | `src/snake_ai/` | FastAPI server: runs the games and streams them over a WebSocket (`/ws/play`), plus HTTP routes under `/api` |
| Frontend | `frontend/` | Vite + React + TypeScript web app that shows and controls the games |
| Local GUI | `src/snake_ai/local_main.py` | Tkinter window that plays (or trains) without a server |
| Models | `policy_models/` | Trained policies (`*.pt`) and the catalog `models.json` |
| Deployment | `Dockerfile`, `frontend/Dockerfile`, `k8s/` | Container images and Kubernetes manifests |

**Contents:** [Requirements](#requirements) · [Install](#install) · [Run locally](#run-locally) · [Configuration](#configuration) · [Docker](#docker) · [Kubernetes](#kubernetes) · [TODO](#todo)

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs a matching Python, ≥ 3.12)
- Node.js with npm, for the frontend
- Docker, for the images; `kubectl` plus `gcloud` or minikube, for Kubernetes

## Install

```bash
uv sync                       # backend, with the dev tools and the CUDA build of torch
cd frontend && npm install    # frontend
```

On a machine without an NVIDIA GPU, the CPU-only build of torch is much smaller:

```bash
uv sync --no-group gpu --group cpu
```

`uv run` syncs the environment by itself before each command, but only with the default groups (`dev`, `gpu`). After installing the CPU group, use `uv run --no-sync …`, or the next `uv run` switches back to the GPU build.

## Run locally

### Web app (backend + frontend)

Start the backend in one terminal:

```bash
uv run uvicorn snake_ai.main:main_app --reload
```

and the frontend dev server in a second one:

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. The dev server forwards `/ws` and `/api` to the backend on port 8000, so both have to be running. Code changes reload automatically.

Other commands:

```bash
uv run ruff check src                              # lint the backend (--fix also sorts imports)
npm run build                                      # in frontend/: type-check and bundle into dist/
npm run lint                                       # in frontend/: lint with oxlint
uv run python -m websockets ws://127.0.0.1:8000/ws/play   # raw WebSocket test client
```

More frontend details are in [frontend/README.md](frontend/README.md).

### Local GUI

```bash
uv run python -m snake_ai.local_main [--model FILE] [--train] [--seed N]
```

| Flag | Default | Effect |
|---|---|---|
| `--model FILE` | `dqn-1m.pt` | Model to play with: a file name in the model directory (`policy_models/`), or an absolute path |
| `--train` | off | Train a new policy first, save it as `<timestamp>_model_<steps>.pt` in the model directory, then play with it |
| `--seed N` | `42` | Seed for torch, the food placement and training, so runs are reproducible |

Examples:

```bash
uv run python -m snake_ai.local_main                         # play with dqn-1m.pt
uv run python -m snake_ai.local_main --model dqn-200k.pt     # play with another model
uv run python -m snake_ai.local_main --train --seed 7        # train, then play
```

## Configuration

The backend (and the local GUI) read `SNAKE_AI_*` environment variables. Relative paths are resolved against the current directory.

| Variable | Default | Meaning |
|---|---|---|
| `SNAKE_AI_LOG_LEVEL` | `INFO` | Log level (upper case) |
| `SNAKE_AI_LOG_PATH` | `<repo>/logs/app.log` | Log file |
| `SNAKE_AI_MODEL_PATH` | `<repo>/policy_models` | Model directory |
| `SNAKE_AI_MODEL_CATALOG` | `<model path>/models.json` | Catalog of the models the server offers |
| `SNAKE_AI_MAX_GAME_COUNT` | `50` | Maximum number of games at once; the oldest one is closed when a new one exceeds it |

## Docker

Both images listen on port 8080 and run as a non-root user.

```bash
docker build -t <image-name>:<version> .                     # backend, with policy_models/ baked in
docker build -t <image-name>-frontend:<version> frontend     # frontend (npm run build, served by nginx)
docker push <image-name>:<version>                           # optional, for a cluster that pulls from Docker Hub
```

Like the Vite dev server, the frontend's nginx forwards `/ws` and `/api` to the backend, so the app keeps talking to its own origin. The backend address is read from `BACKEND_URL` when the container starts (default `http://backend:8080`). To try the frontend image against a backend running locally with uvicorn on port 8000:

```bash
docker run --rm -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  <image-name>-frontend:<version>
```

Then open http://localhost:8080. nginx fails to start if the host in `BACKEND_URL` can't be resolved. A new backend route that the frontend calls needs a `location` in `frontend/nginx/default.conf.template` and an entry in `frontend/vite.config.ts`.

## Kubernetes

The manifests in `k8s/` deploy the images `muellern99/snake-ai-backend` and `muellern99/snake-ai-frontend` from Docker Hub. The Ingress and `backend_config.yaml` use GKE features, so minikube runs without them.

### GKE

```bash
export PROJECT_ID=my-project
export REGION=europe-west3
gcloud config set project $PROJECT_ID

# Create an Autopilot cluster and point kubectl at it
gcloud container clusters create-auto my-cluster --region=$REGION
gcloud container clusters get-credentials my-cluster --region=$REGION

kubectl apply -f k8s/
kubectl get ingress backend-ingress -w     # wait for the external IP
```

It can take a few minutes after the IP shows up before the application is reachable.

### minikube

The frontend's nginx already forwards `/ws` and `/api` to the backend, so the Ingress is left out:

```bash
minikube start
kubectl apply -f k8s/backend_deploy.yaml -f k8s/backend_svc.yaml \
              -f k8s/frontend_deploy.yaml -f k8s/frontend_svc.yaml
minikube service frontend-svc              # opens the frontend in the browser
```

To use images you built locally instead of the ones on Docker Hub, build them inside minikube with the tags from the manifests (they use `imagePullPolicy: IfNotPresent`):

```bash
minikube image build -t muellern99/snake-ai-backend:1.0.1 .
minikube image build -t muellern99/snake-ai-frontend:1.0.0 frontend
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

- [x] 🔴 Run the setup on a Kubernetes cluster.
  - [x] 🟡 Dockerfiles: one for the backend (uv, with the models baked into the image or mounted from a volume), and one for the frontend (`npm run build`, served by nginx), unless FastAPI serves `frontend/dist` itself.
  - [x] 🟡 Manifests (or a Helm chart): a Deployment and a Service for the backend, plus an Ingress that routes `/ws` to the backend with WebSocket support and `/` to the frontend. Pass the `SNAKE_AI_*` settings through a ConfigMap.
  - [x] 🟢 Add a health endpoint for the liveness and readiness probes.
- [ ] 🔴 Implement the Redis-backed registry: `save` and `delete` are stubs and the Redis code is commented out.
  - Games live in the server process's memory, so start with one backend replica. With more replicas, reconnecting by `game_id` only works with sticky sessions or the Redis-backed registry.

### Testing

- [ ] 🟡 Add tests. There are none yet. Good first targets are `SnakeEnv` (moves, wrapping, eating, winning, losing), `parse_message`, and the `GameRunner` lifecycle (attach, detach, expire).
