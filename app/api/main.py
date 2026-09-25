"""
FastAPI Backend for GridForecast
Provides REST API for forecasting, scenarios, alerts, and backtesting.

Endpoints are plain `def` (not `async def`) so FastAPI runs the CPU-bound
model work in its threadpool instead of blocking the event loop.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..inference import (
    CHANNEL_DESCRIPTIONS,
    CHECKPOINT_DESCRIPTIONS,
    CHECKPOINT_SPECS,
    HORIZONS,
    PAPER_RESULTS,
    REPORTED_RESULTS,
    SEQ_LEN,
    MAX_HORIZON,
    ForecastEngine,
)

# Global engine instance (loaded once at startup)
engine: Optional[ForecastEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Starting GridForecast API...")
    engine = ForecastEngine(device="cpu")
    print("GridForecast API ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="GridForecast API",
    description="Transformer load & oil-temperature forecasting with FM-LLM",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ForecastRequest(BaseModel):
    window_idx: int = Field(..., ge=0, description="Test window index")
    horizon: int = Field(..., description="Forecast horizon in hours", ge=96, le=720)
    checkpoint: str = Field("instnorm", description="Model checkpoint to use")
    context_hours: int = Field(168, ge=0, le=SEQ_LEN, description="Hours of history to return for plotting")


class ScenarioRequest(BaseModel):
    window_idx: int = Field(..., ge=0)
    horizon: int = Field(..., ge=96, le=720)
    checkpoint: str = Field("instnorm")
    perturbations: Dict[str, float] = Field(
        ...,
        description="Fractional change to the last 96h of each channel, e.g. {'HUFL': 0.2, 'OT': -0.1}",
    )


class BacktestRequest(BaseModel):
    checkpoint: str = Field("instnorm")
    num_windows: int = Field(50, ge=1, le=200)
    horizons: List[int] = Field(default=list(HORIZONS))


def _engine() -> ForecastEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine


def _check_checkpoint(eng: ForecastEngine, checkpoint: str):
    if checkpoint not in eng.available_checkpoints():
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint '{checkpoint}' not available. Available: {eng.available_checkpoints()}",
        )


def _check_window(eng: ForecastEngine, window_idx: int):
    if not 0 <= window_idx < eng.data.num_windows:
        raise HTTPException(status_code=400, detail=f"window_idx must be < {eng.data.num_windows}")


def _check_horizon(horizon: int):
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {list(HORIZONS)}")


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=409, detail=str(e))


# API Endpoints
@app.get("/health")
def health():
    eng = _engine()
    return {
        "status": "ok",
        "device": eng.device,
        "available_windows": eng.data.num_windows,
        "checkpoints": eng.available_checkpoints(),
        "live_checkpoints": eng.live_checkpoints(),
        "precomputed": {k: len(v) for k, v in eng.precomputed.items()},
    }


@app.get("/checkpoints")
def list_checkpoints():
    eng = _engine()
    return {
        "checkpoints": [
            {
                "name": name,
                "filename": CHECKPOINT_SPECS[name][0],
                "description": CHECKPOINT_DESCRIPTIONS[name],
                "live": eng.is_live(name),
                "precomputed_windows": len(eng.precomputed.get(name, {})),
            }
            for name in eng.available_checkpoints()
        ]
    }


@app.get("/channels")
def list_channels():
    eng = _engine()
    return {
        "channels": eng.get_channel_names(),
        "descriptions": CHANNEL_DESCRIPTIONS,
    }


@app.get("/windows")
def list_windows(checkpoint: str = Query("instnorm")):
    eng = _engine()
    _check_checkpoint(eng, checkpoint)
    windows = eng.windows(checkpoint)
    return {
        "windows": windows,
        "start_dates": [str(eng.window_start(w)) for w in windows],
        "precomputed": sorted(eng.precomputed.get(checkpoint, {})),
    }


@app.get("/results")
def reported_results():
    """Full-test-set rollout results (experiment 10) and the paper's Table A.12."""
    return {
        "checkpoints": {k: {str(h): v for h, v in r.items()} for k, r in REPORTED_RESULTS.items()},
        "paper": {str(h): v for h, v in PAPER_RESULTS.items()},
    }


@app.post("/forecast")
def forecast(request: ForecastRequest):
    eng = _engine()
    _check_checkpoint(eng, request.checkpoint)
    _check_window(eng, request.window_idx)
    _check_horizon(request.horizon)

    result = _run(
        eng.forecast, request.window_idx, request.horizon, request.checkpoint, request.context_hours
    )
    for key in ("history", "predictions", "ground_truth"):
        result[key] = result[key].tolist()
    return result


@app.post("/forecast/all_horizons")
def forecast_all_horizons(window_idx: int = Query(...), checkpoint: str = Query("instnorm")):
    eng = _engine()
    _check_checkpoint(eng, checkpoint)
    _check_window(eng, window_idx)
    metrics = _run(eng.horizon_metrics, window_idx, checkpoint)
    return {"window_idx": window_idx, "results": {str(h): m for h, m in metrics.items()}}


@app.post("/scenario")
def scenario(request: ScenarioRequest):
    eng = _engine()
    _check_checkpoint(eng, request.checkpoint)
    _check_window(eng, request.window_idx)
    _check_horizon(request.horizon)

    valid_channels = set(eng.get_channel_names())
    for ch in request.perturbations:
        if ch not in valid_channels:
            raise HTTPException(status_code=400, detail=f"Invalid channel: {ch}")

    result = _run(
        eng.scenario_forecast,
        request.window_idx,
        request.horizon,
        request.checkpoint,
        request.perturbations,
    )
    for key in ("base", "predictions"):
        result[key] = result[key].tolist()
    return result


@app.post("/backtest")
def backtest(request: BacktestRequest):
    eng = _engine()
    _check_checkpoint(eng, request.checkpoint)
    for h in request.horizons:
        _check_horizon(h)
    out = _run(eng.backtest, request.checkpoint, request.num_windows, tuple(request.horizons))
    return {"results": {str(h): r for h, r in out["results"].items()}, "windows": out["windows"]}


@app.get("/demo/window/{window_idx}")
def get_demo_window(window_idx: int):
    """Raw (real-unit) history and future for a test window."""
    eng = _engine()
    _check_window(eng, window_idx)
    start = window_idx + SEQ_LEN
    return {
        "window_idx": window_idx,
        "history_dates": eng.data.test_dates[window_idx:start].astype(str).tolist(),
        "history": eng.data.test_raw[window_idx:start].tolist(),
        "future_dates": eng.data.test_dates[start:start + MAX_HORIZON].astype(str).tolist(),
        "future": eng.data.test_raw[start:start + MAX_HORIZON].tolist(),
        "channels": eng.get_channel_names(),
    }


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000)
