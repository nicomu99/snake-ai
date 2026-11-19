export interface ModelOption {
  id: string
  name: string
}

// TODO: fetch from `GET /models` once the backend has that endpoint
// (see the TODO list in the README). Must match policy_models/models.json.
export const MODELS: ModelOption[] = [
  { id: 'dqn-200k', name: 'DQN 200K' },
  { id: 'dqn-1m', name: 'DQN 1M' },
]

/** Tiles per side. Must match TILE_COUNT in src/snake_ai/utils/constants.py. */
export const BOARD_SIZE = 5

/** tick_ms sent to the server for each speed level (level 1 = index 0). */
export const SPEED_LEVELS = [1000, 600, 400, 250, 150, 100, 60, 30, 15, 5]
export const DEFAULT_SPEED_LEVEL = 5
