"""
The forecast API (docs/api_contract.md), served from the responses LiveService keeps in memory.

    uvicorn service.app:app --port 7860                  # locally, live OpenAQ updates
    SERVICE_FETCH=0 uvicorn service.app:app --port 7860  # no fetching: serve the downloaded history

The endpoints live on `router`, so they can be attached to any FastAPI app. On the Hugging Face
Space, app.py (written by scripts/deploy_space.py) lets Gradio start the server, as ZeroGPU
requires, and attaches this router to Gradio's app. Every response carries the open CORS header
itself, so the browser app can call it whichever server is running it.
"""

import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from service.live import LiveService

service = LiveService(fetch=os.environ.get("SERVICE_FETCH", "1") != "0")
router = APIRouter()


def reply(body, status: int = 200) -> JSONResponse:
    """
    Args:
        body: JSON-serialisable response.
        status: HTTP status code.

    Returns:
        The response, readable from any website (CORS open, as the contract says).
    """
    return JSONResponse(body, status_code=status, headers={"Access-Control-Allow-Origin": "*"})


def error(status: int, text: str) -> JSONResponse:
    """
    Args:
        status: HTTP status code.
        text: what went wrong.

    Returns:
        The contract's error response: {"error": text}.
    """
    return reply({"error": text}, status)


def not_ready():
    """Returns: a 503 response while the first forecasts are still being made (or None once ready)."""
    if service.live is None:
        return error(503, service.error or "starting up: the first forecasts take a few minutes after a restart")
    return None


def health_body() -> dict:
    """Returns: the /v1/health body: the contract's fields, or status "starting" / "error" before the first forecasts."""
    if service.live is None:
        return {"status": service.status, "error": service.error}
    body = dict(service.live["health"])
    if service.error:
        body["error"] = service.error
    return body


@router.get("/v1/health")
def health():
    """Returns: GET /v1/health (503 until the first forecasts are ready)."""
    return reply(health_body(), 200 if service.live is not None else 503)


@router.get("/v1/sensors")
def sensors():
    """Returns: GET /v1/sensors."""
    return not_ready() or reply(service.live["sensors"])


@router.get("/v1/forecasts")
def forecasts():
    """Returns: GET /v1/forecasts."""
    return not_ready() or reply(service.live["forecasts"])


@router.get("/v1/forecasts/{sensor_id}")
def forecast_detail(sensor_id: str):
    """
    Args:
        sensor_id: OpenAQ sensor id.

    Returns:
        GET /v1/forecasts/{sensor_id}; 404 for an unknown sensor or one without a forecast this hour.
    """
    waiting = not_ready()
    if waiting:
        return waiting
    found = service.detail(sensor_id)
    if found is not None:
        return reply(found)
    if service.known(sensor_id):
        return error(404, f"sensor {sensor_id} has no forecast this hour (no recent reading, or too many gaps)")
    return error(404, f"unknown sensor {sensor_id}")


@router.get("/v1/replay")
def replay(date: str):
    """
    Args:
        date: "YYYY-MM-DD".

    Returns:
        GET /v1/replay; 400 for a date without enough data around it.
    """
    if service.raw is None or service.model is None:
        return error(503, "starting up")
    try:
        return reply(service.replay(date))
    except ValueError as e:
        return error(400, str(e))


@asynccontextmanager
async def lifespan(_app):
    """Start loading and the hourly loop in the background, so the server answers at once."""
    service.start()
    yield


app = FastAPI(title="Accra PM2.5 forecast API", version="0.1", lifespan=lifespan)
app.include_router(router)
