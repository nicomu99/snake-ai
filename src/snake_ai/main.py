"""Entry point for the Snake AI WebSocket server.

Builds the FastAPI application, wires up the WebSocket router and manages
the lifetime of the shared policy, the game runner registry and the sweeper
that removes abandoned games.

Example:
    Start the server (from any directory; paths come from the settings,
    see ``utils.settings``)::

        $ uv run uvicorn snake_ai.main:main_app --reload

Attributes:
    main_app (FastAPI): The ASGI application to pass to an ASGI server.
"""

from contextlib import asynccontextmanager

# import redis.asyncio as redis
from fastapi import FastAPI

from snake_ai.ai import Policy
from snake_ai.api import GameRunnerRegistry
from snake_ai.api.game_runner_sweeper import GameRunnerSweeper
from snake_ai.api.sockets import router
from snake_ai.utils import configure_logging, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Sets up and tears down the shared server state.

    On startup, loads ``model.pt`` from the model directory
    (``Settings.model_path``) as a ``Policy`` that every game shares,
    creates the game runner registry and starts the sweeper. The policy and
    the registry are stored on ``app.state`` (``policy``,
    ``runner_registry``). On shutdown, stops the sweeper and closes every
    running game.

    Args:
        app: The FastAPI application whose state is populated.

    Yields:
        None. Control returns to FastAPI while the server is running.

    Raises:
        FileNotFoundError: If ``model.pt`` does not exist in the model
            directory. The server then fails to start.
    """
    # redis_url = "redis://localhost:6379"      # TODO: url
    settings = get_settings()
    policy = Policy.load_model("model.pt", model_path=settings.model_path)
    app.state.policy = policy
    # app.state.redis = redis.Redis.from_url(redis_url, decode_responses=True)
    app.state.runner_registry = GameRunnerRegistry(app.state.policy)
    # , app.state.redis)
    game_session_sweeper = GameRunnerSweeper(app.state.runner_registry)
    game_session_sweeper.start()
    try:
        yield
    finally:
        await game_session_sweeper.stop()
        await app.state.runner_registry.close_all()


def main():
    """Configures logging and builds the FastAPI application.

    This does not start a server; pass the returned app to an ASGI server
    such as uvicorn.

    Returns:
        FastAPI: The configured application with the WebSocket router
        included.
    """
    configure_logging()

    app = FastAPI(title="Snake AI", lifespan=lifespan)
    app.include_router(router)
    return app


main_app = main()
