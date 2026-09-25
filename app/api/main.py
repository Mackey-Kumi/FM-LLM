"""
FastAPI Backend for GridForecast
Provides REST API for forecasting, scenarios, alerts, and backtesting.
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
import uvicorn
import threading
import time

from ..inference import (
    ForecastEngine, 
    CHECKPOINT_SPECS, 
    HORIZONS, 
    CHANNEL_NAMES,
    prepare_demo_data,
)

# Global engine instance (loaded once at startup)
engine: Optional[ForecastEngine] = None
engine_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle"""
    global engine
    # Startup
    print("Starting GridForecast API...")
    prepare_demo_data()
    engine = ForecastEngine(checkpoint_name="instnorm", device="cpu", use_cache=True)
    print("GridForecast API ready!")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="GridForecast API",
    description="Electricity Demand Forecasting with FM-LLM",
    version="0.1.0",
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


class ForecastResponse(BaseModel):
    predictions: List[List[float]]  # [horizon, channels]
    ground_truth: List[List[float]]
    channels: List[str]
    horizon: int
    window_idx: int
    metrics: Dict[str, float]


class ScenarioRequest(BaseModel):
    window_idx: int = Field(..., ge=0)
    horizon: int = Field(..., ge=96, le=720)
    perturbations: Dict[str, float] = Field(
        ..., 
        description="Channel perturbations, e.g. {'OT': 0.05, 'HUFL': -0.1}"
    )


class ScenarioResponse(BaseModel):
    predictions: List[List[float]]
    channels: List[str]
    perturbations: Dict[str, float]


class BacktestRequest(BaseModel):
    num_windows: int = Field(50, ge=1, le=200)
    horizons: List[int] = Field(default=list(HORIZONS))


class BacktestResponse(BaseModel):
    results: Dict[str, Dict[str, float]]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    available_windows: int


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health():
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return HealthResponse(
        status="ok",
        model=engine.checkpoint_name,
        device=engine.device,
        available_windows=engine.get_available_windows(),
    )


@app.get("/checkpoints")
async def list_checkpoints():
    available = []
    for name, (filename, _) in CHECKPOINT_SPECS.items():
        available.append({
            "name": name,
            "filename": filename,
            "description": f"FM-LLM {'with' if CHECKPOINT_SPECS[name][1] else 'without'} instance norm"
        })
    return {"checkpoints": available}


@app.get("/channels")
async def list_channels():
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"channels": engine.get_channel_names()}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Switch checkpoint if needed
    if request.checkpoint != engine.checkpoint_name:
        with engine_lock:
            engine = ForecastEngine(checkpoint_name=request.checkpoint, device="cpu", use_cache=True)
    
    max_windows = engine.get_available_windows()
    if request.window_idx >= max_windows:
        raise HTTPException(
            status_code=400, 
            detail=f"window_idx must be < {max_windows}"
        )
    
    if request.horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon must be one of {list(HORIZONS)}"
        )
    
    try:
        result = engine.forecast_window(request.window_idx, request.horizon)
        
        # Compute metrics
        preds = result["predictions"]
        truth = result["ground_truth"]
        mse = float(np.mean((preds - truth) ** 2))
        mae = float(np.mean(np.abs(preds - truth)))
        
        return ForecastResponse(
            predictions=preds.tolist(),
            ground_truth=truth.tolist(),
            channels=result["channels"],
            horizon=result["horizon"],
            window_idx=result["window_idx"],
            metrics={"mse": mse, "mae": mae},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast/all_horizons")
async def forecast_all_horizons(window_idx: int = Query(...), checkpoint: str = Query("instnorm")):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if checkpoint != engine.checkpoint_name:
        with engine_lock:
            engine = ForecastEngine(checkpoint_name=checkpoint, device="cpu", use_cache=True)
    
    max_windows = engine.get_available_windows()
    if window_idx >= max_windows:
        raise HTTPException(status_code=400, detail=f"window_idx must be < {max_windows}")
    
    try:
        results = engine.forecast_all_horizons(window_idx)
        # Convert to JSON-serializable
        output = {}
        for h, r in results.items():
            output[str(h)] = {
                "predictions": r["predictions"].tolist(),
                "ground_truth": r["ground_truth"].tolist(),
                "mse": float(r["mse"]),
                "mae": float(r["mae"]),
                "channels": r["channels"],
            }
        return {"window_idx": window_idx, "results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario", response_model=ScenarioResponse)
async def scenario(request: ScenarioRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    max_windows = engine.get_available_windows()
    if request.window_idx >= max_windows:
        raise HTTPException(status_code=400, detail=f"window_idx must be < {max_windows}")
    
    # Validate channel names
    valid_channels = set(engine.get_channel_names())
    for ch in request.perturbations:
        if ch not in valid_channels:
            raise HTTPException(status_code=400, detail=f"Invalid channel: {ch}")
    
    try:
        result = engine.scenario_forecast(
            request.window_idx, 
            request.horizon, 
            request.perturbations
        )
        return ScenarioResponse(
            predictions=result["predictions"].tolist(),
            channels=result["channels"],
            perturbations=result["perturbations"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Run in background thread to avoid blocking
        result = engine.backtest(request.num_windows, tuple(request.horizons))
        return BacktestResponse(results=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo/window/{window_idx}")
async def get_demo_window(window_idx: int):
    """Get raw test window data for preview"""
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    max_windows = engine.get_available_windows()
    if window_idx >= max_windows:
        raise HTTPException(status_code=400, detail=f"window_idx must be < {max_windows}")
    
    history = engine.test_norm[window_idx : window_idx + 672]
    future = engine.test_norm[window_idx + 672 : window_idx + 672 + 720]
    
    # Denormalize for display
    hist_display = history * CHANNEL_STD + CHANNEL_MEAN
    fut_display = future * CHANNEL_STD + CHANNEL_MEAN
    
    return {
        "window_idx": window_idx,
        "history": hist_display.tolist(),
        "future": fut_display.tolist(),
        "channels": CHANNEL_NAMES,
    }


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)