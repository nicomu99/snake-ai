# frontend/ notes

- The app always connects to its own origin. Every backend route the frontend calls needs a proxy entry in BOTH `vite.config.ts` (dev; otherwise Vite answers it with `index.html`) and `nginx/default.conf.template` (prod, proxies to `$BACKEND_URL`).
- `src/protocol.ts` mirrors the server's message types — keep it in sync with `src/snake_ai/api/`.
- `src/config.ts` `BOARD_SIZE` must match the backend's `TILE_COUNT`.
- The server doesn't report whether a game is running, so `running` is tracked on the client. `useSnakeGame.ts` re-sends the speed after every `init_message` and handles close codes 4000 (opened in another tab) / 4001 (too many games).
