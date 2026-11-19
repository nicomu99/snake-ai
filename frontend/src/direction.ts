import { BOARD_SIZE } from './config'
import type { Pos, Snapshot } from './protocol'

/** A unit step [dx, dy]; y grows downwards. */
export type Dir = [number, number]

/** Direction from `a` to the neighbouring tile `b`, taking the wrap into account. */
export function dirBetween(a: Pos, b: Pos): Dir {
  const n = BOARD_SIZE
  const wrap = (d: number) => (d === n - 1 ? -1 : d === -(n - 1) ? 1 : d)
  return [wrap(b[0] - a[0]), wrap(b[1] - a[1])]
}

/**
 * The direction of the snake's last move after `snap`, or null before the
 * first move.
 *
 * Snapshots don't include it, so it is derived from the neck → head step, or
 * for a one-tile snake from the head's previous position.
 */
export function nextHeadDirection(
  prevDir: Dir | null,
  prevHead: Pos | null,
  snap: Snapshot,
): Dir | null {
  if (snap.step_count === 0) return null
  if (snap.snake.length > 1) return dirBetween(snap.snake[1], snap.snake[0])
  if (prevHead) {
    const d = dirBetween(prevHead, snap.snake[0])
    if (Math.abs(d[0]) + Math.abs(d[1]) === 1) return d
  }
  return prevDir
}
