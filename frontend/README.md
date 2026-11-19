# Snake AI frontend

A Vite + React + TypeScript client for the Snake AI WebSocket server.

```bash
npm install
npm run dev      # http://localhost:5173, needs the backend on :8000
npm run build    # type-check and bundle into dist/
npm run lint     # oxlint
```

Start the backend first (from the repo root):

```bash
uv run uvicorn snake_ai.main:main_app --reload
```

The Vite dev server proxies `/ws` to `localhost:8000` (see `vite.config.ts`),
so the app always connects to its own origin.

- `src/protocol.ts` — message types, mirrors `src/snake_ai/api/`.
- `src/useSnakeGame.ts` — the WebSocket connection, reconnects and game state.
- `src/Board.tsx` — draws the board on a small canvas that CSS scales up.
- `src/config.ts` — model list (hard-coded until `GET /models` exists), board
  size and speed levels.
