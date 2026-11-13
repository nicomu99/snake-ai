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
