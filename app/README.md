# GridForecast App

**Transformer overheating early warning** built on FM-LLM
(Frequency-Enhanced Mixture-of-Experts for Time Series Forecasting). FM-LLM
reads 28 days of hourly data from a power transformer (ETTh1), each signal
from its own history, and forecasts up to 30 days ahead. GridForecast turns
that into operator warnings: *will the oil temperature cross its alarm
level in the next few days, and how much notice do we get?*

## Features

- **Early Warning** (main screen): daily outlook with NORMAL / WARNING status, time to breach,
  a 30-day risk strip and a suggested operator response. "Reveal what actually happened"
  shows whether the warning was right.
- **Warning Performance**: replays one outlook per day across the test period and scores
  FM-LLM's warnings (hit rate, false alarms, CSI, lead time) against the simple rules an
  operator could use without a model: hold the last value, repeat yesterday, repeat last week.
- **What-if**: scale the monitored signal's last 4 days and see how the outlook responds (live mode).
  FM-LLM forecasts each channel from its own history only (channel independence), so load
  changes don't affect the oil-temperature forecast.
- **Forecast Explorer**: all channels, all horizons, against actuals
- **Model & Accuracy**: architecture, checkpoint comparison, results vs. the paper, backtest

Alarm level, early-warning margin and outlook length are site settings in
the sidebar. ETTh1's test period is winter (Oct–Feb), when oil temperature
peaks around 15 °C, so the demo defaults to a 10 °C alarm level. Summer
data in the training period passes 40 °C, and a real site would set limits
seasonally.

## How forecasts are served

| Mode | Needs | Speed |
|------|-------|-------|
| **Precomputed** | `app/precomputed/forecasts.npz` only | Instant, fully offline |
| **Live** | checkpoint `.pt` files in `kaggle/datasets/checkpoint/` + `HF_TOKEN` (Llama-3.2-1B) | ~seconds per forecast on CPU; Llama loads on first use |

A window is served from the precomputed file when it's there, and runs live
otherwise. The Scenario Simulator always runs live. **For a demo, use
precomputed forecasts** so nothing depends on downloads or the network.

## Quick Start

All commands run from the **repo root**.

### 1. Install dependencies

```bash
uv pip install -r app/requirements.txt   # or: pip install -r app/requirements.txt
```

### 2. Get forecasts to serve (one-time)

**Option A: Kaggle GPU (recommended, ~1h on T4)**: runs every test window for all
four checkpoints, keeps one window per day (91 windows), and re-prints the full
test-set MSE/MAE as a check against experiment 10:

```bash
cd kaggle/experiments/11_app_forecasts
kaggle kernels push -p .
kaggle kernels status mackeykumi/fm-llm-etth1-app-forecasts   # until COMPLETE
kaggle kernels output mackeykumi/fm-llm-etth1-app-forecasts -p ./output -o
mkdir -p ../../../app/precomputed && cp output/forecasts.npz ../../../app/precomputed/
```

**Option B: locally** (needs the checkpoint `.pt` files + `HF_TOKEN`; slower on CPU):

```bash
python -m app.precompute_forecasts                      # all checkpoints, every 24th window
python -m app.precompute_forecasts --checkpoints instnorm --stride 48   # quicker
```

For live mode (scenarios), also copy the checkpoint `.pt` files into
`kaggle/datasets/checkpoint/` and put `HF_TOKEN=...` in `.env`.

### 3. Run the application

```bash
python -m app.main          # starts the API, waits until it's healthy, then the UI

# or separately:
python -m uvicorn app.api.main:app --port 8000
streamlit run app/ui/main.py
```

Then open **http://localhost:8501**.

## Project Structure

```
app/
├── main.py                  # Entry point (runs API + UI)
├── inference.py             # Engine: data, model, precomputed + live forecasts
├── precompute_forecasts.py  # Build app/precomputed/forecasts.npz locally
├── requirements.txt
├── api/main.py              # FastAPI backend
├── ui/main.py               # Streamlit UI (5 tabs)
├── demo_data/               # ETTh1 CSV (the app reads this, no network needed)
└── precomputed/             # forecasts.npz (generated)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
│  │ Warning │ │Performance│ │ What-if  │ │Explore │ │ Model  │  │
│  └────┬────┘ └─────┬─────┘ └────┬─────┘ └───┬────┘ └───┬────┘  │
└───────┼─────────────┼─────────────┼───────────┼──────────┼───────┘
        │             │             │           │          │
        ▼             ▼             ▼           ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  /outlook  /warning/evaluate  /scenario  /forecast  /backtest  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Engine                             │
│  precomputed forecasts.npz (instant)  OR  live FM-LLM rollout:  │
│  FourierEmbeddingModule → Llama-3.2-1B → FANMoEDecoder          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status, available checkpoints, live vs. precomputed |
| `/checkpoints` | GET | Available model checkpoints |
| `/channels` | GET | Channel names and descriptions |
| `/windows` | GET | Forecast windows (and start dates) for a checkpoint |
| `/results` | GET | Full-test-set results (experiment 10) and paper Table A.12 |
| `/forecast` | POST | Single-horizon forecast with history, actuals, metrics |
| `/forecast/all_horizons` | POST | MSE/MAE at every horizon for one window |
| `/scenario` | POST | Base vs. perturbed forecast (live only) |
| `/backtest` | POST | Walk-forward backtest |
| `/outlook` | POST | Early-warning status for one issue time |
| `/warning/evaluate` | POST | Replay daily outlooks, score FM-LLM vs. naive baselines |
| `/accuracy/compare` | POST | Forecast error of FM-LLM vs. naive baselines |
| `/demo/window/{idx}` | GET | Raw test window data |

MSE/MAE are reported on the globally normalized scale, the same as the
paper's Table A.12. Charts and downloads use real units.

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
| `OT` | Oil temperature (target) |
| `HUFL` | High useful load |
| `HULL` | High useless load |
| `MUFL` | Middle useful load |
| `MULL` | Middle useless load |
| `LUFL` | Low useful load |
| `LULL` | Low useless load |

## Troubleshooting

- **"Nothing to serve"**: generate `app/precomputed/forecasts.npz` (step 2) or add checkpoints.
- **"Gated repo" / 401**: `.env` needs an `HF_TOKEN` with access to `meta-llama/Llama-3.2-1B` (live mode only).
- **Scenario tab says precomputed only**: the checkpoint `.pt` files aren't in `kaggle/datasets/checkpoint/`.
- **UI says API not running**: the API is still loading; refresh after a few seconds.

## Development

```bash
python -m pytest tests/test_app.py   # engine + API tests, no GPU/Llama/checkpoints needed
python test_engine.py                # smoke test against your real forecasts/checkpoints
```

API docs: http://localhost:8000/docs
