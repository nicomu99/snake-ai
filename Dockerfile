FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV SNAKE_AI_MODEL_PATH=/models \
    SNAKE_AI_LOG_PATH=/tmp/snake-ai/app.log

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["uv", "run", "uvicorn", "app.main:main_app", "--host", "0.0.0.0", "--port", "8080"]