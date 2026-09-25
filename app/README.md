# GridForecast App

Electricity Demand Forecasting Application built on FM-LLM (Frequency-Enhanced Mixture-of-Experts for Time Series Forecasting).

## Features

- **Live Forecast Dashboard**: Generate forecasts for 96h, 192h, 336h, 720h horizons
- **Scenario Simulator**: Modify input channels (temperature, load) and see forecast impact
- **Backtest & Metrics**: Walk-forward evaluation with comparison to paper benchmarks
- **Alerts & Monitoring**: Threshold and anomaly detection on forecasts
- **Model Explorer**: Architecture details and checkpoint comparison

## Quick Start

### 1. Install Dependencies

```bash
cd app
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

### 2. Prepare Backbone Cache (One-time, ~10-20 min)

This downloads Llama-3.2-1B (requires HF_TOKEN in `.env`) and pre-computes backbone outputs for all test windows.

```bash
python -m app.precache_backbone
```

> **Note**: Requires `HF_TOKEN` in `.env` with access to `meta-llama/Llama-3.2-1B`. The model is gated on Hugging Face.

### 3. Run the Application

```bash
# Option A: Run both API and UI together
python -m app.main

# Option B: Run separately
# Terminal 1: API
python -m uvicorn app.api.main:app --reload --port 8000

# Terminal 2: UI
streamlit run app/ui/main.py
```

Then open: **http://localhost:8501**

## Project Structure

```
app/
├── main.py                 # Entry point (runs API + UI)
├── requirements.txt        # Dependencies
├── README.md              # This file
├── inference.py           # Core inference engine (FM-LLM + cached backbone)
├── precache_backbone.py   # Pre-compute backbone outputs
├── download_model.py      # Download GGUF model (alternative)
├── test_engine.py         # Quick test script
├── api/
│   ├── __init__.py
│   └── main.py            # FastAPI backend
├── ui/
│   ├── __init__.py
│   └── main.py            # Streamlit UI (5 tabs)
├── demo_data/             # Cached ETTh1 data for offline demo
├── cache/                 # Backbone output cache (generated)
└── models/                # GGUF models (if using llama-cpp)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
│  │Forecast │ │ Scenarios │ │ Backtest │ │ Alerts │ │ Model  │  │
│  └────┬────┘ └─────┬─────┘ └────┬─────┘ └───┬────┘ └───┬────┘  │
└───────┼─────────────┼─────────────┼───────────┼──────────┼───────┘
        │             │             │           │          │
        ▼             ▼             ▼           ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  /forecast  /forecast/all_horizons  /scenario  /backtest       │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Engine                             │
│  FourierEmbeddingModule → Cached Llama Backbone → FANMoEDecoder │
│                          ↓                                       │
│              Pre-computed cache (instant)                       │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status |
| `/checkpoints` | GET | Available model checkpoints |
| `/channels` | GET | Channel names |
| `/forecast` | POST | Single horizon forecast |
| `/forecast/all_horizons` | POST | All horizons at once |
| `/scenario` | POST | Scenario with perturbations |
| `/backtest` | POST | Walk-forward backtest |
| `/demo/window/{idx}` | GET | Raw test window data |

## Model Checkpoints

| Name | Description | Instance Norm |
|------|-------------|---------------|
| `instnorm` | Best model (instance norm + dropout) | ✅ |
| `dropout_longer` | Dropout, longer training | ❌ |
| `dropout` | Dropout only | ❌ |
| `no_dropout` | Baseline | ❌ |

## ETTh1 Channels

| Channel | Description |
|---------|-------------|
| `OT` | Oil Temperature (target) |
| `HUFL` | High Useful Load |
| `HULL` | High Useless Load |
| `MUFL` | Medium Useful Load |
| `MULL` | Medium Useless Load |
| `LUFL` | Low Useful Load |
| `LULL` | Low Useless Load |

## Performance Notes

- **First run**: Downloads Llama-3.2-1B (~2.5GB) + builds cache (~10-20 min)
- **Subsequent runs**: Instant inference (<1s per forecast) using cached backbone
- **Memory**: ~2GB RAM for model + cache
- **Device**: CPU only (no GPU required)

## Troubleshooting

### "Gated repo" / 401 Error
Ensure `.env` contains valid `HF_TOKEN` with access to `meta-llama/Llama-3.2-1B`.

### "No backbone cache found"
Run `python -m app.precache_backbone` first.

### Slow inference
Cache not loaded. Check `app/cache/backbone_outputs.pkl` exists.

### Import errors
Run `uv pip install -r requirements.txt` from `app/` directory.

## Development

### Run tests
```bash
python test_engine.py
```

### API docs
Open http://localhost:8000/docs when API is running.

### Add new checkpoints
Place `.pt` files in `../kaggle/datasets/checkpoint/` and update `CHECKPOINT_SPECS` in `inference.py`.

## License

Part of FM-LLM reproduction project.