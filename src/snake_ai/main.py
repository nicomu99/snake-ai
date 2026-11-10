from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI

from snake_ai.api.sockets import router
from snake_ai.ai import Agent
from snake_ai.utils import configure_logging
from snake_ai.registries import GameSessionRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = "redis://localhost:6379"      # TODO: url
    agent = Agent()
    agent.load_model()
    app.state.agent = agent
    app.state.redis = redis.Redis.from_url(redis_url, decode_responses=True)
    app.state.game_sessions = GameSessionRegistry(app.state.redis, app.state.agent)
    yield


def main():
    configure_logging()

    app = FastAPI(title="Snake AI", lifespan=lifespan)
    app.include_router(router)
    return app


main_app = main()
