import pytest

from snake_ai.snake import SnakeEnv
from snake_ai.utils import TILE_COUNT

LAST = TILE_COUNT - 1
FAR_AWAY = [TILE_COUNT // 2, TILE_COUNT // 2]


@pytest.fixture
def env():
    """A fresh env for every test, so state can't leak between tests."""
    return SnakeEnv(seed=42)


# --- reset / setup ---------------------------------------------------------


def test_reset(env):
    env.perform_move("RIGHT")
    env.reset()
    assert env.snake_position == [[0, 0]]
    assert env.last_snake_position == [0, 0]
    assert env.direction is None
    assert not env.lost
    assert not env.won
    assert env.score == 0


def test_same_seed_places_food_the_same():
    assert SnakeEnv(seed=1).price_position == SnakeEnv(seed=1).price_position


def test_reset_with_seed_reseeds(env):
    env.reset(seed=7)
    first = env.price_position
    env.reset(seed=7)
    assert env.price_position == first


def test_food_is_on_board_and_not_on_snake(env):
    for seed in range(50):
        env.reset(seed=seed)
        x, y = env.price_position
        assert 0 <= x < TILE_COUNT and 0 <= y < TILE_COUNT
        assert env.price_position not in env.snake_position


# --- movement ----------------------------------------------------------------


@pytest.mark.parametrize(
    "start, direction, expected",
    [
        ([3, 3], "UP", [3, 2]),
        ([3, 3], "DOWN", [3, 4]),
        ([3, 3], "LEFT", [2, 3]),
        ([3, 3], "RIGHT", [4, 3]),
        # wrapping around the edges (torus)
        ([0, 0], "LEFT", [LAST, 0]),
        ([0, 0], "UP", [0, LAST]),
        ([LAST, LAST], "RIGHT", [0, LAST]),
        ([LAST, LAST], "DOWN", [LAST, 0]),
    ],
)
def test_head_moves(env, start, direction, expected):
    env.snake_position = [start]
    env.price_position = [1, 6]  # not on any expected tile
    env.perform_move(direction)
    assert env.snake_position == [expected]
    assert env.direction == direction


def test_body_follows_head(env):
    env.snake_position = [[3, 3], [2, 3], [1, 3]]
    env.direction = "RIGHT"
    env.price_position = FAR_AWAY
    env.perform_move("DOWN")
    assert env.snake_position == [[3, 4], [3, 3], [2, 3]]
    assert env.last_snake_position == [1, 3]


def test_normal_step_reward(env):
    env.price_position = FAR_AWAY
    reward, _, lost, won = env.perform_move("RIGHT")
    assert reward == pytest.approx(-0.001)
    assert not lost and not won
    assert env.get_game_state() == "pending"


def test_invalid_direction_raises(env):
    with pytest.raises(ValueError):
        env.perform_move("SIDEWAYS")


# --- direction reversal --------------------------------------------------------


@pytest.mark.parametrize("direction", ["UP", "DOWN", "LEFT", "RIGHT"])
def test_any_first_move_is_valid(env, direction):
    assert not env.is_invalid_move(direction)


@pytest.mark.parametrize(
    "current, reverse",
    [("UP", "DOWN"), ("DOWN", "UP"), ("LEFT", "RIGHT"), ("RIGHT", "LEFT")],
)
def test_reversing_loses(env, current, reverse):
    env.price_position = FAR_AWAY
    env.perform_move(current)
    before = [list(p) for p in env.snake_position]

    reward, _, lost, won = env.perform_move(reverse)

    assert reward == -2
    assert lost and not won
    assert env.snake_position == before  # the snake doesn't move
    assert env.get_game_state() == "direction_reversal"


# --- eating ---------------------------------------------------------------------


def test_eating_grows_snake_and_scores(env):
    env.price_position = [1, 0]
    reward, _, lost, won = env.perform_move("RIGHT")

    assert reward == 2
    assert env.score == 1
    assert env.snake_position == [[1, 0], [0, 0]]  # grew into old tail
    assert not lost and not won
    assert env.price_position not in env.snake_position


# --- self collision -------------------------------------------------------------


def test_self_collision_loses(env):
    # Head at [1, 1] came from the right; moving DOWN hits its own body.
    env.snake_position = [[1, 1], [2, 1], [2, 2], [1, 2], [0, 2]]
    env.direction = "LEFT"
    env.price_position = FAR_AWAY

    reward, _, lost, won = env.perform_move("DOWN")

    assert reward == -2
    assert lost and not won
    assert env.get_game_state() == "collision"


# --- winning --------------------------------------------------------------------


def test_filling_the_board_wins(env):
    all_tiles = [[x, y] for x in range(TILE_COUNT) for y in range(TILE_COUNT)]
    others = [t for t in all_tiles if t not in ([0, 0], [1, 0])]
    env.snake_position = [[0, 0], *others]  # every tile except [1, 0]
    env.price_position = [1, 0]

    reward, _, lost, won = env.perform_move("RIGHT")

    assert reward == 10
    assert won and not lost
    assert len(env.snake_position) == TILE_COUNT * TILE_COUNT
    assert env.get_game_state() == "game_won"


# --- game over ------------------------------------------------------------------


def test_moves_after_game_over_do_nothing(env):
    env.game_over()
    before = [list(p) for p in env.snake_position]

    reward, _, lost, _ = env.perform_move("RIGHT")

    assert reward == 0
    assert lost
    assert env.snake_position == before


# --- helpers --------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([0, 0], [0, 0], 0),
        ([0, 0], [3, 4], 5),
        ([0, 0], [LAST, 0], 1),  # shorter way round the edge
        ([0, 0], [LAST, LAST], 2**0.5),
    ],
)
def test_wrapped_distance(a, b, expected):
    assert SnakeEnv.get_wrapped_distance(a, b) == pytest.approx(expected)


def test_get_state_is_not_changed_by_later_moves(env):
    env.price_position = FAR_AWAY
    snake, _, _ = env.get_state()
    env.perform_move("RIGHT")
    assert snake == [[0, 0]]
