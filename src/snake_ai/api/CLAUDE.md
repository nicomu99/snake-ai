# api/ notes

- `GameRunnerRegistry` keys are `game_runner:<id>`. Connecting without a `game_id`, or with an unknown one, creates a new runner; a known id re-attaches. When a new runner exceeds `max_game_count`, the oldest is evicted even if its client is connected (close code 4001 "too many games").
- `GameRunner`: only one sink can be attached at a time — a new attach closes the old socket with code 4000 ("opened in another tab"). If a tick raises, the loop survives: it pauses, sets `crashed`, sends `game_crashed` (also on every attach until reset), and ignores `start`/`resume` until a `reset`.
- The server doesn't report whether a game is running; the client tracks it.
- `GET /ready` returns 503 until the lifespan has stored the registry.
