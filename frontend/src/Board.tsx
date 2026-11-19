import { useEffect, useRef } from 'react'
import { BOARD_SIZE } from './config'
import { type Dir, dirBetween } from './direction'
import type { Pos, Snapshot } from './protocol'

/** Art pixels per tile. The canvas is tiny and scaled up by CSS. */
const T = 8
const N = BOARD_SIZE

const COLORS = {
  tileA: '#0f1a0e',
  tileB: '#132111',
  head: '#9dff6a',
  neck: [57, 230, 42], // body color right behind the head (RGB)…
  tail: [27, 110, 22], // …fading to this at the tail
  dead: '#ff3b3b',
  eye: '#07090a',
}

const FOOD_SPRITE = [
  '....s...',
  '...sl...',
  '.rrrrrr.',
  'rrwrrrrr',
  'rwrrrrrr',
  'rrrrrrrd',
  '.rrrrrd.',
  '..rr.rr.',
]
const FOOD_COLORS: Record<string, string> = {
  r: '#ff3b3b',
  d: '#b8202a',
  w: '#ffd6d6',
  s: '#8a5a2b',
  l: '#39e62a',
}

/** Fades from neck to tail, so a tile keeps nearly the same color between steps. */
function bodyColor(index: number, length: number): string {
  const t = length > 2 ? (index - 1) / (length - 2) : 0
  const mix = COLORS.neck.map((c, k) => Math.round(c + (COLORS.tail[k] - c) * t * 0.7))
  return `rgb(${mix.join(' ')})`
}

function drawSprite(
  ctx: CanvasRenderingContext2D,
  sprite: string[],
  palette: Record<string, string>,
  x: number,
  y: number,
) {
  sprite.forEach((row, dy) => {
    for (let dx = 0; dx < row.length; dx++) {
      const color = palette[row[dx]]
      if (!color) continue
      ctx.fillStyle = color
      ctx.fillRect(x + dx, y + dy, 1, 1)
    }
  })
}

/** Fills the 1px gap on the side of `pos` that faces `dir`. */
function fillEdge(ctx: CanvasRenderingContext2D, pos: Pos, dir: Dir) {
  const x = pos[0] * T
  const y = pos[1] * T
  if (dir[0] === 1) ctx.fillRect(x + T - 1, y + 1, 1, T - 2)
  if (dir[0] === -1) ctx.fillRect(x, y + 1, 1, T - 2)
  if (dir[1] === 1) ctx.fillRect(x + 1, y + T - 1, T - 2, 1)
  if (dir[1] === -1) ctx.fillRect(x + 1, y, T - 2, 1)
}

function draw(ctx: CanvasRenderingContext2D, snap: Snapshot | null, headDir: Dir) {
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      ctx.fillStyle = (x + y) % 2 ? COLORS.tileA : COLORS.tileB
      ctx.fillRect(x * T, y * T, T, T)
    }
  }
  if (!snap) return

  drawSprite(ctx, FOOD_SPRITE, FOOD_COLORS, snap.food[0] * T, snap.food[1] * T)

  const { snake } = snap
  // Body from tail to neck, then the head on top (it overlaps on a collision).
  for (let i = snake.length - 1; i >= 1; i--) {
    const seg = snake[i]
    ctx.fillStyle = bodyColor(i, snake.length)
    ctx.fillRect(seg[0] * T + 1, seg[1] * T + 1, T - 2, T - 2)
    const toFront = dirBetween(seg, snake[i - 1])
    fillEdge(ctx, seg, toFront)
    fillEdge(ctx, snake[i - 1], [-toFront[0], -toFront[1]])
  }

  const [hx, hy] = [snake[0][0] * T, snake[0][1] * T]
  ctx.fillStyle = snap.lost ? COLORS.dead : COLORS.head
  ctx.fillRect(hx + 1, hy + 1, T - 2, T - 2)
  if (snake.length > 1) fillEdge(ctx, snake[0], dirBetween(snake[0], snake[1]))

  // Eyes on the side the head is facing.
  const [dx, dy] = headDir
  const eyes: Pos[] =
    dx === 1 ? [[5, 2], [5, 5]]
    : dx === -1 ? [[2, 2], [2, 5]]
    : dy === -1 ? [[2, 2], [5, 2]]
    : [[2, 5], [5, 5]]
  ctx.fillStyle = COLORS.eye
  for (const [ex, ey] of eyes) ctx.fillRect(hx + ex, hy + ey, 1, 1)
}

interface BoardProps {
  snapshot: Snapshot | null
  headDir: Dir | null // null before the first move
}

export function Board({ snapshot, headDir }: BoardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    draw(ctx, snapshot, headDir ?? [1, 0])
  }, [snapshot, headDir])

  return (
    <canvas
      ref={canvasRef}
      className="board"
      width={N * T}
      height={N * T}
      aria-label="Snake board"
    />
  )
}
