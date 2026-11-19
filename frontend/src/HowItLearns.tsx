import { BOARD_SIZE } from './config'
import type { Dir } from './direction'
import type { Snapshot } from './protocol'

// All numbers below mirror the backend: the reward shaping in
// snake/snake_env.py (perform_move), the network in ai/policy_model.py, the
// state encoding in ai/policy.py and the hyperparameters in
// ai/dqn_trainer.py. Update this file when those change.

const DIRECTIONS: { label: string; dir: Dir }[] = [
  { label: 'UP', dir: [0, -1] },
  { label: 'DOWN', dir: [0, 1] },
  { label: 'LEFT', dir: [-1, 0] },
  { label: 'RIGHT', dir: [1, 0] },
]

const REWARDS: { event: string; value: string; good: boolean }[] = [
  { event: 'FILL THE BOARD (WIN)', value: '+100', good: true },
  { event: 'EAT THE APPLE', value: '+20', good: true },
  { event: 'MOVE CLOSER TO APPLE', value: '+1', good: true },
  { event: 'MOVE AWAY FROM APPLE', value: '-1', good: false },
  { event: 'EVERY STEP', value: '-0.1', good: false },
  { event: 'REVERSE (GAME OVER)', value: '-10', good: false },
  { event: 'BITE ITSELF (GAME OVER)', value: '-20', good: false },
]

const LAYERS = [
  { name: 'INPUT', size: '7×5×5' },
  { name: 'CONV 3×3', size: '64×3×3' },
  { name: 'CONV 3×1', size: '64×3×1' },
  { name: 'DENSE', size: '128' },
  { name: 'Q-VALUES', size: '4' },
]

/** One 5×5 input channel; `lit(x, y)` says which cells are 1. */
function Channel({ label, lit }: { label: string; lit: (x: number, y: number) => boolean }) {
  const cells = []
  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      cells.push(<i key={`${x},${y}`} className={lit(x, y) ? 'on' : ''} />)
    }
  }
  return (
    <figure className="channel">
      <div className="channel-grid" style={{ gridTemplateColumns: `repeat(${BOARD_SIZE}, 1fr)` }}>
        {cells}
      </div>
      <figcaption>{label}</figcaption>
    </figure>
  )
}

interface Props {
  snapshot: Snapshot | null
  headDir: Dir | null
}

export function HowItLearns({ snapshot, headDir }: Props) {
  const snake = snapshot?.snake ?? []
  const at = (x: number, y: number) => (p: [number, number]) => p[0] === x && p[1] === y
  const head = snake[0]

  return (
    <section className="info" aria-labelledby="info-title">
      <h2 id="info-title">HOW IT LEARNS</h2>
      <p>
        No one taught this snake the rules. It is a neural network trained with{' '}
        <em>reinforcement learning</em>: it played the game over and over, got points for good
        moves and penalties for bad ones, and slowly learned which moves pay off.
      </p>

      <h3>1. WHAT IT SEES</h3>
      <p>
        Each step the board becomes 7 layers of 5×5 ones and zeros. This is the live input for
        the game on the left:
      </p>
      <div className="channels">
        <Channel label="HEAD" lit={(x, y) => !!head && at(x, y)(head)} />
        <Channel label="BODY" lit={(x, y) => snake.slice(1).some(at(x, y))} />
        <Channel label="APPLE" lit={(x, y) => !!snapshot && at(x, y)(snapshot.food)} />
        {DIRECTIONS.map(({ label, dir }) => (
          <Channel
            key={label}
            label={label}
            lit={() => !!headDir && headDir[0] === dir[0] && headDir[1] === dir[1]}
          />
        ))}
      </div>
      <p className="small">
        The last four layers say which way the snake is heading: the layer for the current
        direction is all ones, the others all zeros.
      </p>

      <h3>2. THE BRAIN</h3>
      <p>
        A small convolutional network turns those layers into four <em>Q-values</em>, one per
        move. A Q-value estimates the total future reward of making that move now. The snake
        always picks the highest.
      </p>
      <ol className="net">
        {LAYERS.map((l) => (
          <li key={l.name}>
            <strong>
              {l.name}
              <span>{l.size}</span>
            </strong>
          </li>
        ))}
      </ol>

      <h3>3. THE REWARDS</h3>
      <p>Points it gets for each move during training:</p>
      <table className="rewards">
        <tbody>
          {REWARDS.map((r) => (
            <tr key={r.event}>
              <td>{r.event}</td>
              <td className={r.good ? 'good' : 'bad'}>{r.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small">
        The board wraps around at the edges, so there are no walls to hit. The only way to lose
        is biting itself or trying to turn back.
      </p>

      <h3>4. THE TRAINING</h3>
      <p>
        The method is <em>Double DQN</em> (Deep Q-Network). The models in the picker are named
        after how many moves they trained for.
      </p>
      <ul className="steps">
        <li>
          <strong>EXPLORE.</strong> It starts with 100% random moves. After every game the
          chance of a random move shrinks by 1%, down to 10%.
        </li>
        <li>
          <strong>REMEMBER.</strong> Every move goes into a memory of the last 10,000 moves:
          what it saw, what it did, the reward and what happened next.
        </li>
        <li>
          <strong>LEARN.</strong> After the first 10,000 moves, every 4th move it replays 256
          random memories and nudges its Q-values toward
          <code>reward + 0.95 × best future Q-value</code>
        </li>
        <li>
          <strong>STAY STABLE.</strong> The future Q-value comes from a frozen copy of the
          network, refreshed every 1,000 moves. The live network picks the best next move and the
          copy scores it. That is the “double” in Double DQN, and it stops the network from
          overrating its own guesses.
        </li>
      </ul>
    </section>
  )
}
