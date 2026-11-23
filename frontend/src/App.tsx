import { useEffect, useState, type ReactNode } from 'react'
import { Board } from './Board'
import { BOARD_SIZE, DEFAULT_SPEED_LEVEL, SPEED_LEVELS } from './config'
import { HowItLearns } from './HowItLearns'
import { useModels } from './useModels'
import { useSnakeGame } from './useSnakeGame'

const pad = (n: number, width: number) => String(n).padStart(width, '0')

const LOSS_REASONS: Record<string, string> = {
  collision: 'BIT ITSELF',
  direction_reversal: 'TRIED TO REVERSE',
}

export default function App() {
  const [speedLevel, setSpeedLevel] = useState(DEFAULT_SPEED_LEVEL)
  const game = useSnakeGame(SPEED_LEVELS[speedLevel - 1])
  const { status, snapshot, headDir, running, crashed, canPlay, notice } = game

  // Keyboard: Space = play/pause, R = reset.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(target.tagName)) return
      if (e.code === 'Space') {
        e.preventDefault()
        if (running) game.pause()
        else game.play()
      } else if (e.key === 'r' || e.key === 'R') {
        game.reset()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [running, game])

  // Notices disappear after a few seconds.
  const [dismissedId, setDismissedId] = useState<number | null>(null)
  useEffect(() => {
    if (!notice) return
    const t = window.setTimeout(() => setDismissedId(notice.id), 3000)
    return () => window.clearTimeout(t)
  }, [notice])
  const visibleNotice = notice && notice.id !== dismissedId ? notice : null

  let overlay: ReactNode = null
  if (status === 'taken_over') {
    overlay = (
      <>
        <p>OPENED IN ANOTHER TAB</p>
        <button onClick={game.reconnect}>PLAY HERE</button>
      </>
    )
  } else if (status === 'evicted') {
    overlay = (
      <>
        <p>GAME CANCELED</p>
        <p className="small">TOO MANY GAMES ON THE SERVER</p>
        <button onClick={game.reconnect}>NEW GAME</button>
      </>
    )
  } else if (status !== 'open' || !snapshot) {
    overlay = <p className="blink">{status === 'reconnecting' ? 'RECONNECTING' : 'CONNECTING'}...</p>
  } else if (crashed) {
    overlay = (
      <>
        <p className="danger">GAME CRASHED</p>
        <p className="small">PRESS RESET</p>
      </>
    )
  } else if (snapshot.won) {
    overlay = <p className="win blink">YOU WIN!</p>
  } else if (snapshot.lost) {
    overlay = (
      <>
        <p className="danger">GAME OVER</p>
        <p className="small">{LOSS_REASONS[snapshot.game_state] ?? ''}</p>
      </>
    )
  } else if (!running) {
    overlay = <p className="blink">{snapshot.step_count === 0 ? 'PRESS START' : 'PAUSED'}</p>
  }

  const connected = status === 'open'
  const models = useModels(connected)
  // Until the list has loaded (or if it failed), still show the current model.
  const modelOptions =
    models ?? (game.modelId ? [{ id: game.modelId, name: game.modelId }] : [])

  return (
    <div className="layout">
      <main className="cabinet">
        <h1>
          SNAKE<span>.AI</span>
        </h1>

        <div className="hud">
          <Stat label="SCORE" value={pad(snapshot?.score ?? 0, 2)} />
          <Stat label="LENGTH" value={`${pad(snapshot?.snake.length ?? 0, 2)}/${BOARD_SIZE ** 2}`} />
          <Stat label="STEPS" value={pad(snapshot?.step_count ?? 0, 4)} />
        </div>

        <div className={`screen ${overlay ? 'dimmed' : ''}`}>
          <Board snapshot={snapshot} headDir={headDir} />
          {overlay && <div className="overlay">{overlay}</div>}
        </div>

        <div className="buttons">
          {running ? (
            <button onClick={game.pause} disabled={!connected}>
              ❚❚ PAUSE
            </button>
          ) : (
            <button onClick={game.play} disabled={!canPlay}>
              ▶ PLAY
            </button>
          )}
          <button onClick={game.reset} disabled={!connected}>
            ↺ RESET
          </button>
        </div>

        <div className="settings">
          <label>
            <span>MODEL</span>
            <select
              value={game.modelId ?? ''}
              onChange={(e) => game.setModel(e.target.value)}
              disabled={!connected || !models}
            >
              {modelOptions.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>SPEED</span>
            <input
              type="range"
              min={1}
              max={SPEED_LEVELS.length}
              value={speedLevel}
              onChange={(e) => setSpeedLevel(Number(e.target.value))}
            />
            <output>{pad(speedLevel, 2)}</output>
          </label>
        </div>

        <p className="hint">SPACE PLAY/PAUSE · R RESET</p>

        <footer>
          <span className={`led ${connected ? 'on' : ''}`} />
          {connected ? 'ONLINE' : 'OFFLINE'}
          {game.gameId && <span className="game-id">#{game.gameId}</span>}
        </footer>

        {visibleNotice && (
          <div key={visibleNotice.id} className={`notice ${visibleNotice.kind}`} role="status">
            {visibleNotice.text}
          </div>
        )}
      </main>
      <HowItLearns snapshot={snapshot} headDir={headDir} />
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
