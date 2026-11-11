from contextlib import asynccontextmanager

# import redis.asyncio as redis
from fastapi import FastAPI

from snake_ai.ai import Agent
from snake_ai.api import GameRunnerRegistry
from snake_ai.api.game_runner_sweeper import GameRunnerSweeper
from snake_ai.api.sockets import router
from snake_ai.utils import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # redis_url = "redis://localhost:6379"      # TODO: url
    agent = Agent()
    agent.load_model()
    app.state.agent = agent
    # app.state.redis = redis.Redis.from_url(redis_url, decode_responses=True)
    app.state.runner_registry = GameRunnerRegistry(app.state.agent)
    # , app.state.redis)
    game_session_sweeper = GameRunnerSweeper(app.state.runner_registry)
    game_session_sweeper.start()
    try:
        yield
    finally:
        await game_session_sweeper.stop()
        await app.state.runner_registry.close_all()


def main():
    configure_logging()

    app = FastAPI(title="Snake AI", lifespan=lifespan)
    app.include_router(router)
    return app


main_app = main()
