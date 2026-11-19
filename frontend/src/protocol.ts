// Message types of the WebSocket protocol at /ws/play.
// Mirrors src/snake_ai/api/game_runner.py (server → client) and
// src/snake_ai/api/command_message.py (client → server).

/** A board position as [x, y]; y grows downwards. */
export type Pos = [number, number]

export type GameState = 'pending' | 'game_won' | 'collision' | 'direction_reversal'

/** The game view from `Game.snapshot()`. */
export interface Snapshot {
  snake: Pos[] // head first
  food: Pos
  score: number
  lost: boolean
  won: boolean
  game_state: GameState
  step_count: number
}

export type ServerMessage =
  | (Snapshot & { type: 'snapshot' })
  | (Snapshot & { type: 'init_message'; game_id: string; model_id: string })
  | {
      type: 'end_message'
      reason: GameState
      score: number
      step_count: number
      snake_length: number
      won: boolean
    }
  | { type: 'model_changed'; message: string; model_id: string }
  | { type: 'game_crashed'; message: string }
  | { type: 'game_canceled'; message: string }
  | { type: 'error'; message: string }

export type ClientCommand =
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'reset'; seed?: number }
  | { type: 'speed'; tick_ms: number }
  | { type: 'set_model'; model_id: string }

/** WebSocket close codes the server uses. */
export const CLOSE_TAKEN_OVER = 4000 // "opened in another tab"
export const CLOSE_EVICTED = 4001 // "too many games"

export function toSnapshot(msg: Snapshot): Snapshot {
  const { snake, food, score, lost, won, game_state, step_count } = msg
  return { snake, food, score, lost, won, game_state, step_count }
}
