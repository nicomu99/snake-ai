import { useCallback, useEffect, useRef, useState } from 'react'
import { type Dir, nextHeadDirection } from './direction'
import {
  CLOSE_EVICTED,
  CLOSE_TAKEN_OVER,
  type ClientCommand,
  type Pos,
  type ServerMessage,
  type Snapshot,
  toSnapshot,
} from './protocol'

export type ConnectionStatus =
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'taken_over' // the game was opened in another tab (close code 4000)
  | 'evicted' // the server dropped the game, too many games (close code 4001)

export interface Notice {
  id: number
  text: string
  kind: 'info' | 'error'
}

// sessionStorage: a reload re-attaches to the same game, while a new tab
// starts its own game instead of stealing this one.
const GAME_ID_KEY = 'snake-ai:game-id'

function readGameId(): string | null {
  try {
    return sessionStorage.getItem(GAME_ID_KEY)
  } catch {
    return null
  }
}

function writeGameId(id: string | null) {
  try {
    if (id === null) sessionStorage.removeItem(GAME_ID_KEY)
    else sessionStorage.setItem(GAME_ID_KEY, id)
  } catch {
    // Storage unavailable (private mode etc.): reloads just start a new game.
  }
}

function socketUrl(gameId: string | null): string {
  const url = new URL('/ws/play', window.location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  if (gameId) url.searchParams.set('game_id', gameId)
  return url.toString()
}

/**
 * Connects to the game server and exposes the game state plus commands.
 *
 * The server doesn't report whether the loop is running, so `running` is
 * tracked locally: it is set by our play/pause commands and cleared by any
 * (re)init, end, crash or disconnect.
 */
export function useSnakeGame(tickMs: number) {
  const [status, setStatus] = useState<ConnectionStatus>('connecting')
  const [gameId, setGameId] = useState<string | null>(null)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [headDir, setHeadDir] = useState<Dir | null>(null)
  const [modelId, setModelId] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [crashed, setCrashed] = useState(false)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [session, setSession] = useState(0) // bump to force a reconnect

  const wsRef = useRef<WebSocket | null>(null)
  const tickMsRef = useRef(tickMs)

  const send = useCallback((cmd: ClientCommand): boolean => {
    const ws = wsRef.current
    if (ws?.readyState !== WebSocket.OPEN) return false
    ws.send(JSON.stringify(cmd))
    return true
  }, [])

  const notify = useCallback((text: string, kind: Notice['kind'] = 'info') => {
    setNotice({ id: Date.now(), text, kind })
  }, [])

  useEffect(() => {
    let disposed = false
    let socket: WebSocket | null = null
    let attempt = 0
    let dir: Dir | null = null
    let prevHead: Pos | null = null

    function showSnapshot(snap: Snapshot) {
      dir = nextHeadDirection(dir, prevHead, snap)
      prevHead = snap.snake[0]
      setSnapshot(snap)
      setHeadDir(dir)
    }

    function handle(ws: WebSocket, msg: ServerMessage) {
      switch (msg.type) {
        case 'init_message':
          writeGameId(msg.game_id)
          setGameId(msg.game_id)
          setModelId(msg.model_id)
          showSnapshot(toSnapshot(msg))
          setRunning(false) // attach and reset always leave the game paused
          setCrashed(false) // a game_crashed message follows if it still is
          // The server starts every game at its own default speed.
          ws.send(JSON.stringify({ type: 'speed', tick_ms: tickMsRef.current }))
          break
        case 'snapshot':
          showSnapshot(toSnapshot(msg))
          break
        case 'end_message':
          setRunning(false)
          break
        case 'model_changed':
          setModelId(msg.model_id)
          notify('MODEL LOADED')
          break
        case 'game_crashed':
          setRunning(false)
          setCrashed(true)
          break
        case 'game_canceled':
          setRunning(false)
          break
        case 'error':
          notify(msg.message, 'error')
          break
      }
    }

    function connect() {
      setStatus(attempt === 0 ? 'connecting' : 'reconnecting')
      const ws = new WebSocket(socketUrl(readGameId()))
      socket = ws

      ws.onopen = () => {
        attempt = 0
        wsRef.current = ws
        setStatus('open')
      }
      ws.onmessage = (event) => {
        let msg: ServerMessage
        try {
          msg = JSON.parse(event.data)
        } catch {
          return
        }
        handle(ws, msg)
      }
      ws.onclose = (event) => {
        if (wsRef.current === ws) wsRef.current = null
        if (disposed) return
        setRunning(false)
        if (event.code === CLOSE_TAKEN_OVER) {
          setStatus('taken_over')
        } else if (event.code === CLOSE_EVICTED) {
          writeGameId(null)
          setStatus('evicted')
        } else {
          setStatus('reconnecting')
          const delay = Math.min(500 * 2 ** attempt, 8000)
          attempt += 1
          timer = window.setTimeout(connect, delay)
        }
      }
    }

    // Deferred so React StrictMode's dev-only mount → unmount → mount cancels
    // the first attempt before it opens a socket (two sockets for one game
    // would kick each other out with close code 4000).
    let timer = window.setTimeout(connect, 0)

    return () => {
      disposed = true
      window.clearTimeout(timer)
      socket?.close()
      wsRef.current = null
    }
  }, [session, notify])

  // Keep the server in sync with the speed slider.
  useEffect(() => {
    tickMsRef.current = tickMs
    send({ type: 'speed', tick_ms: tickMs })
  }, [tickMs, send])

  const over = !!snapshot && (snapshot.lost || snapshot.won)
  const canPlay = status === 'open' && !!snapshot && !over && !crashed

  const play = useCallback(() => {
    if (!canPlay || !snapshot) return
    if (send({ type: snapshot.step_count === 0 ? 'start' : 'resume' })) setRunning(true)
  }, [canPlay, snapshot, send])

  const pause = useCallback(() => {
    if (send({ type: 'pause' })) setRunning(false)
  }, [send])

  const reset = useCallback(() => {
    send({ type: 'reset' })
  }, [send])

  const setModel = useCallback(
    (id: string) => {
      send({ type: 'set_model', model_id: id })
    },
    [send],
  )

  const reconnect = useCallback(() => setSession((s) => s + 1), [])

  return {
    status,
    gameId,
    snapshot,
    headDir,
    modelId,
    running,
    crashed,
    over,
    canPlay,
    notice,
    play,
    pause,
    reset,
    setModel,
    reconnect,
  }
}
