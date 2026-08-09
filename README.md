# FM-LLM Reproduction

Reproduction of *FM-LLM: A frequency-enhanced mixture-of-experts framework for
adapting LLMs to time series forecasting* (Gu et al., Knowledge-Based Systems,
2026): Fourier Embedding Module, FAN-MoE decoder with load-balanced routing,
hybrid time-frequency loss.

## Status

ETTh1 (input=672, predict=96) is trained and evaluated end-to-end:

- Full training run (batch=256, matching the paper's Table 2) on Kaggle's
  T4x2, 16 epochs, early-stopped at best val_loss=1.0228.
- Real test-set evaluation: **MSE=0.4911, MAE=0.4729** vs. the paper's
  0.342/0.380 (Table A.12) — right ballpark for a first pass, no dropout or
  hyperparameter tuning yet.

Next: add dropout (paper Table 2 specifies 0.2, currently missing), try
gradient checkpointing (paper reports ~6GB training memory on a single GPU;
we needed ~12-15GB per GPU on Kaggle's T4x2 — checkpointing might let this
run on one GPU instead of two), re-measure, and only then extend to the
remaining 10 benchmark datasets.

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
