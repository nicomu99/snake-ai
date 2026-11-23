"""The HTTP routes: the model list and the Kubernetes probes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from snake_ai.ai import PolicySelector
from snake_ai.api.deps import get_policy_selector
from snake_ai.api.schemas import ModelItem, ModelResponse

router = APIRouter()


@router.get("/models")
async def retrieve_models(
    policy_selector: Annotated[PolicySelector, Depends(get_policy_selector)],
) -> ModelResponse:
    """Lists the models that were loaded at startup.

    Models listed in the catalog whose file was missing are left out, so
    every returned id can be passed to ``set_model``.

    Args:
        policy_selector: The shared selector holding the loaded models.

    Returns:
        The default model id and the loaded models, in catalog order.
    """
    default_model_id = policy_selector.default_model_id
    models = [
        ModelItem(id=mid, name=m_entry.name)
        for mid, m_entry in policy_selector.model_catalog.items()
    ]
    return ModelResponse(default_model_id=default_model_id, models=models)


@router.get("/health")
async def health():
    """Answers the liveness probe.

    Checks no dependencies on purpose. The handler runs on the event loop, so
    the probe times out if the loop is blocked.

    Returns:
        ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """Answers the readiness probe.

    The server is ready once the ``lifespan`` has stored the game runner
    registry on ``app.state``. uvicorn only accepts requests after the
    lifespan startup has finished, so in practice the 503 branch is only
    reached without a lifespan (e.g. a ``TestClient`` used outside a ``with``
    block). It is the place for real checks, e.g. a Redis ping once the
    Redis-backed registry exists.

    Args:
        request: The current request, used to reach ``app.state``.

    Returns:
        ``{"status": "ready"}``, or a 503 response with
        ``{"status": "starting"}`` while the registry is missing.
    """
    game_runner_registry = getattr(request.app.state, "runner_registry", None)
    if game_runner_registry is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "starting"}
        )
    return {"status": "ready"}
