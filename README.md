# FM-LLM Reproduction

Reproduction of *FM-LLM: A frequency-enhanced mixture-of-experts framework for
adapting LLMs to time series forecasting* (Gu et al., Knowledge-Based Systems,
2026): Fourier Embedding Module, FAN-MoE decoder with load-balanced routing,
hybrid time-frequency loss.

## Status

ETTh1 (input=672, predict=96) is trained and evaluated end-to-end, now with
dropout added and autoregressive rollout covering all four paper horizons:

- **Dropout (Table 2: 0.2)** was missing from the first pass; adding it and
  retraining from scratch (`05_dropout_retrain`) improved best val_loss from
  1.0228 to **0.9589**, and it ran the full 30-epoch ceiling without
  early-stopping (vs. `03`'s early stop at epoch 16) — dropout let the model
  keep improving instead of overfitting.
- 96-step test-set evaluation (`04_test_eval`, no-dropout checkpoint):
  MSE=0.4911, MAE=0.4729 vs. the paper's 0.342/0.380 (Table A.12).
- **Autoregressive rollout** (`06_rollout_eval`) now reaches 192/336/720
  steps by feeding the model's own predictions back in as input (sliding
  7-token context), matching how the paper itself reaches these horizons.
  Dropout wins at every horizon:

  | Horizon | No-dropout (MSE/MAE) | Dropout (MSE/MAE) | Paper (Table A.12) |
  |---|---|---|---|
  | 192 | 0.5089 / 0.4882 | 0.4417 / 0.4571 | 0.377 / 0.403 |
  | 336 | 0.5423 / 0.5158 | 0.4858 / 0.4925 | 0.395 / 0.415 |
  | 720 | 0.6381 / 0.5792 | 0.5894 / 0.5661 | 0.397 / 0.429 |

  Dropout narrows the gap to the paper at every horizon but doesn't close
  it — still roughly 15-20% higher MSE than the paper across the board, with
  no further hyperparameter tuning done yet.

Next: investigate what's driving the remaining gap (hyperparameter tuning
beyond Table 2's stated defaults, or re-checking dropout placement against
Fig. 2 more carefully), then extend to the remaining 10 benchmark datasets.
Gradient checkpointing remains out of scope — it isn't paper-mandated, and
the T4x2 `DataParallel` workaround already handles the paper's batch=256.

## Environment

This machine has **no NVIDIA GPU**. All real training/eval runs on Kaggle
(script kernels, pushed via CLI — see [`kaggle/README.md`](kaggle/README.md)
for the full workflow). This machine is for code editing, git, and tiny CPU
smoke tests only.

```powershell
uv sync          # creates .venv/ and installs CPU-only deps
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Select `.venv/Scripts/python.exe` as the VS Code interpreter (already set as
default in `.vscode/settings.json`).

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (Hugging Face) and
`KAGGLE_API_TOKEN` (Kaggle, see `kaggle/README.md`) — `.env` is gitignored,
never commit it.

## Project layout

```
kaggle/
  README.md       # full Kaggle CLI workflow (push/poll/pull, one-time setup)
  datasets/        # private Kaggle datasets we maintain (HF token, checkpoint)
  experiments/      # one script kernel per experiment, numbered in run order
tests/
data/, checkpoints/, outputs/   # gitignored — populated locally or pulled from Kaggle
```

Every experiment's source lives entirely under `kaggle/experiments/` — there
is no separate local `src/` package. Each experiment folder is a
self-contained, faithful port of the paper's equations (Fourier Embedding
Module Eq. 9-11, FAN-MoE decoder Eq. 13-18, hybrid loss Eq. 19-23); later
experiments reuse the same architecture code rather than importing from
earlier ones, since each runs as an independent Kaggle kernel.
