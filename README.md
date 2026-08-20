# FM-LLM Reproduction

Reproduction of *FM-LLM: A frequency-enhanced mixture-of-experts framework for
adapting LLMs to time series forecasting* (Gu et al., Knowledge-Based Systems,
2026): Fourier Embedding Module, FAN-MoE decoder with load-balanced routing,
hybrid time-frequency loss.

## Status

ETTh1 (input=672, predict=96) is trained and evaluated end-to-end. Four
checkpoints have been compared so far — no-dropout, dropout, dropout with a
longer training ceiling, and dropout + instance normalization — and the
last one is decisively the best:

- **Dropout (Table 2: 0.2)** was missing from the first pass; adding it and
  retraining from scratch (`05_dropout_retrain`, then `07` with a longer
  epoch ceiling once `05` turned out to still be improving at cutoff)
  brought best val_loss from 1.0228 down to 0.9540 across two retrains.
- 96-step test-set evaluation (`04_test_eval`, no-dropout checkpoint):
  MSE=0.4911, MAE=0.4729 vs. the paper's 0.342/0.380 (Table A.12).
- **Autoregressive rollout** (`06`/`08`/`10_rollout_eval*`) reaches
  192/336/720 steps by feeding the model's own predictions back in as input
  (sliding 7-token context), matching how the paper itself reaches these
  horizons. That rollout comparison surfaced a real problem: our error grew
  much faster across horizons (+26% MSE, 192→720) than the paper's own
  numbers do (+5%) — the classic symptom of a model not robust to each
  window's own local level/scale.
- **Instance (per-window) normalization** (`09_instance_norm_retrain`), a
  second local normalization layer sitting on top of the existing global
  one — not part of the paper's stated equations, but plausibly inherited
  from PatchTST's channel-independence design (Section 3.2.1), whose own
  architecture is built around exactly this — fixed most of it:

  | Horizon | No-dropout | Dropout | Dropout (longer) | **Instance norm** | Paper (Table A.12) |
  |---|---|---|---|---|---|
  | 192 | 0.5089 / 0.4882 | 0.4417 / 0.4571 | 0.4292 / 0.4445 | **0.3945 / 0.4225** | 0.377 / 0.403 |
  | 336 | 0.5423 / 0.5158 | 0.4858 / 0.4925 | 0.4641 / 0.4730 | **0.4149 / 0.4402** | 0.395 / 0.415 |
  | 720 | 0.6381 / 0.5792 | 0.5894 / 0.5661 | 0.5422 / 0.5350 | **0.4721 / 0.4839** | 0.397 / 0.429 |

  Instance normalization also trained faster and to a better val_loss
  (0.9401, early-stopped at epoch 26) than the longer dropout-only run
  (0.9540, epoch 35). The gap to the paper is now ~5% at 192/336, and down
  from ~37% to ~19% at 720 — real progress, but the paper still stays
  flatter across horizons than we do, so some residual error-accumulation
  gap remains at the longest horizon.

Next: investigate what's still driving the residual 720-horizon gap (the
FAN single-layer ambiguity in Eq. 8, or hyperparameter tuning beyond
Table 2's stated defaults — FAN's `p_ratio` split is our own invented value
and a good first candidate), then extend to the remaining 10 benchmark
datasets. Gradient checkpointing remains out of scope — it isn't
paper-mandated, and the T4x2 `DataParallel` workaround already handles the
paper's batch=256.

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
